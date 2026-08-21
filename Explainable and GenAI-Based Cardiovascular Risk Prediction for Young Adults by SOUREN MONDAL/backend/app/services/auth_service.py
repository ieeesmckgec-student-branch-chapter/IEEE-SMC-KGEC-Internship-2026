from typing import Any
from fastapi import Header, HTTPException
from firebase_admin import auth
from app.services.firebase_service import (
    initialize_firebase,
)
def verify_firebase_token(
    authorization: str | None = Header(
        default=None
    ),
) -> dict[str, Any]:
    """
    Verify the Firebase ID token sent in the
    Authorization request header.
    """

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing.",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=(
                "Authorization header must use the "
                "Bearer token format."
            ),
        )

    token = authorization.removeprefix(
        "Bearer "
    ).strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Firebase ID token is missing.",
        )

    try:
        initialize_firebase()

        decoded_token = auth.verify_id_token(
            token,
            check_revoked=True,
            clock_skew_seconds=60,
        )
        if not decoded_token.get(
            "email_verified",
            False,
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Please verify your email address "
                    "before using this service."
                ),
            )

        return decoded_token

    except auth.RevokedIdTokenError as error:
        raise HTTPException(
            status_code=401,
            detail="The login session has been revoked.",
        ) from error

    except auth.UserDisabledError as error:
        raise HTTPException(
            status_code=403,
            detail="This user account has been disabled.",
        ) from error

    except auth.ExpiredIdTokenError as error:
        raise HTTPException(
            status_code=401,
            detail="The login session has expired.",
        ) from error

    except auth.InvalidIdTokenError as error:
        print("\nFIREBASE INVALID TOKEN ERROR:")
        print(type(error).__name__)
        print(str(error))
        print()

        raise HTTPException(
            status_code=401,
            detail=f"Firebase token error: {error}",
        ) from error

    except Exception as error:
        print(
            "Firebase token verification error:",
            type(error).__name__,
            str(error),
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "Firebase token verification failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error