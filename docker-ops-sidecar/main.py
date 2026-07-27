"""
docker-ops-sidecar

The only container in the honeycomb-net stack that mounts /var/run/docker.sock.
It exposes one narrow HTTP endpoint per `docker exec` operation that api_downlink.py
used to run directly against sibling containers — never a generic exec passthrough.

Not reachable outside honeycomb-net; every request must additionally carry the
X-Internal-Token header matching SIDECAR_SHARED_SECRET.

As each operation gets migrated to a real API (ChirpStack gRPC, Vault HTTP), delete
its endpoint here rather than adding to it — this service is a bridge, not a
permanent fixture.
"""

import json
import re
import subprocess

from fastapi import Depends, FastAPI, Header, HTTPException, Path, status
from pydantic import BaseModel

import config

app = FastAPI(title="docker-ops-sidecar")

SAFE_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9_-]*[a-zA-Z0-9])?$")
SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

SUPERSET_PASSWORD_CHANGE_SCRIPT = """
from superset import create_app
from superset.extensions import db, security_manager
from werkzeug.security import check_password_hash
import sys

email = sys.argv[1]
old_password = sys.argv[2]
new_password = sys.argv[3]

app = create_app()
with app.app_context():
    user = security_manager.find_user(email=email)
    if not user or not check_password_hash(user.password, old_password):
        print('Old password is incorrect')
        sys.exit(1)
    security_manager.reset_password(user.id, new_password)
    db.session.commit()
    print('Password updated')
"""

# Used by the forgot-password flow, where the caller has already been authenticated
# via a one-time reset token instead of the old password — so no old-password check.
SUPERSET_PASSWORD_RESET_SCRIPT = """
from superset import create_app
from superset.extensions import db, security_manager
import sys

email = sys.argv[1]
new_password = sys.argv[2]

app = create_app()
with app.app_context():
    user = security_manager.find_user(email=email)
    if not user:
        print("USER_NOT_FOUND")
        sys.exit(1)
    security_manager.reset_password(user.id, new_password)
    db.session.commit()
    print("PASSWORD_UPDATED")
"""


def require_internal_token(x_internal_token: str = Header(default="")):
    if not config.SIDECAR_SHARED_SECRET or x_internal_token != config.SIDECAR_SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal token")


class AddUserRequest(BaseModel):
    username: str


class SupersetUserCreate(BaseModel):
    username: str
    first_name: str = ""
    last_name: str = ""
    email: str
    password: str
    role: str


class SupersetPasswordChange(BaseModel):
    email: str
    old_password: str
    new_password: str


class SupersetPasswordReset(BaseModel):
    email: str
    new_password: str


def _validate_username(username: str) -> None:
    if "\x00" in username:
        raise HTTPException(status_code=400, detail="Null byte in username is not allowed.")
    if not SAFE_USERNAME_PATTERN.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="Invalid username format. Only letters, digits, '-', '_' are allowed.",
        )


@app.post("/edgex/adduser", dependencies=[Depends(require_internal_token)])
async def edgex_adduser(body: AddUserRequest):
    _validate_username(body.username)
    cmd = [
        "docker", "exec", config.CONTAINER_EDGEX_SECURITY_PROXY,
        "./secrets-config", "proxy", "adduser",
        "--user", body.username,
        "--tokenTTL", "3650d",
        "--jwtTTL", "1d",
        "--useRootToken",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parsed = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse Docker output")
    except subprocess.CalledProcessError as cpe:
        raise HTTPException(status_code=500, detail=f"Docker command failed: {cpe.stderr}")
    return {"password": parsed.get("password", "No password found")}


@app.post("/chirpstack/create-api-key/{name}", dependencies=[Depends(require_internal_token)])
async def chirpstack_create_api_key(name: str = Path(..., min_length=1)):
    if not name.strip() or not SAFE_NAME_PATTERN.match(name):
        raise HTTPException(status_code=400, detail="Invalid or missing 'name' parameter")

    cmd = [
        "docker", "exec", config.CONTAINER_CHIRPSTACK,
        "chirpstack", "--config", "/etc/chirpstack",
        "create-api-key", "--name", name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as cpe:
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {cpe.stderr.strip()}")

    match = re.search(r"token: (\S+)", result.stdout.strip())
    return {"api_key": match.group(1) if match else "No API key found"}


@app.get("/vault/root-token", dependencies=[Depends(require_internal_token)])
async def vault_root_token():
    cmd = ["docker", "exec", config.CONTAINER_VAULT, "cat", config.VAULT_ROOT_PATH]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        parsed = json.loads(output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse JSON from Vault response.")
    except subprocess.CalledProcessError as cpe:
        raise HTTPException(status_code=500, detail=f"Docker command failed: {cpe}")

    root_token = parsed.get("root_token")
    if not root_token:
        raise HTTPException(status_code=404, detail="Root token not found in the JSON file.")
    return {"root_token": root_token}


@app.post("/superset/create-user", dependencies=[Depends(require_internal_token)])
async def superset_create_user(user: SupersetUserCreate):
    cmd = [
        "docker", "exec", config.CONTAINER_SUPERSET,
        "superset", "fab", "create-user",
        "--username", user.username,
        "--firstname", user.first_name,
        "--lastname", user.last_name,
        "--email", user.email,
        "--password", user.password,
        "--role", user.role,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip().lower()
    stderr = result.stderr.strip().lower()

    if "no such container" in stderr or "not found" in stderr:
        raise HTTPException(status_code=404, detail="Superset container or command not found.")
    if "already exists" in stdout or "already exists" in stderr:
        raise HTTPException(status_code=409, detail=f"User with email '{user.email}' already exists.")
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Docker command failed.\nSTDOUT: {stdout}\nSTDERR: {stderr}",
        )
    return {"stdout": result.stdout.strip()}


@app.post("/superset/change-password", dependencies=[Depends(require_internal_token)])
async def superset_change_password(body: SupersetPasswordChange):
    result = subprocess.run(
        [
            "docker", "exec", config.CONTAINER_SUPERSET,
            "python3", "-c", SUPERSET_PASSWORD_CHANGE_SCRIPT,
            body.email, body.old_password, body.new_password,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        if "old password is incorrect" in result.stdout.lower():
            raise HTTPException(status_code=401, detail="Old password is incorrect.")
        raise HTTPException(
            status_code=500,
            detail="Docker exec error: " + (result.stderr.strip() or result.stdout.strip()),
        )

    output = result.stdout.strip()
    if "password updated" not in output.lower():
        raise HTTPException(status_code=500, detail="Unexpected output: " + output)
    return {"stdout": output}


@app.post("/superset/reset-password", dependencies=[Depends(require_internal_token)])
async def superset_reset_password(body: SupersetPasswordReset):
    result = subprocess.run(
        [
            "docker", "exec", config.CONTAINER_SUPERSET,
            "python3", "-c", SUPERSET_PASSWORD_RESET_SCRIPT,
            body.email, body.new_password,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = result.stdout.strip()
    if "PASSWORD_UPDATED" in stdout:
        return {"stdout": stdout}
    if "USER_NOT_FOUND" in stdout:
        raise HTTPException(status_code=404, detail=f"User '{body.email}' not found in Superset.")
    raise HTTPException(status_code=500, detail=f"Unexpected error: {stdout or result.stderr}")
