"""
FitNEase Lobby Load Test Script
================================

Simulates multiple users joining a lobby to test system capacity.

Steps:
1. Register test users via auth API
2. Verify emails directly in DB
3. Login each user to get tokens
4. Join each user to the target group
5. Join each user to the target lobby
6. Report results

Usage:
    # Run on EC2 (not inside Docker — needs both API and DB access)
    python3 load_test_lobby.py --group-id 123 --session-id abc-def-ghi --num-users 24

    # Cleanup test users after testing
    python3 load_test_lobby.py --cleanup
"""

import argparse
import json
import time
import sys
import requests
import mysql.connector
from datetime import datetime


# ============================================================
#  CONFIGURATION
# ============================================================

API_BASE = "http://localhost:8090"  # Nginx gateway on EC2

AUTH_DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3307,   # fitnease-auth-db exposed port
    'database': 'fitnease_auth_db',
    'user': 'root',
    'password': '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU=',
}

TEST_USER_PREFIX = "loadtest_user_"
TEST_PASSWORD = "LoadTest123!"
TEST_EMAIL_DOMAIN = "loadtest.fitnease.local"


# ============================================================
#  HELPERS
# ============================================================

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {msg}")


def api_post(path, data=None, token=None):
    """Make a POST request to the API."""
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{API_BASE}{path}"
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=30)
        return resp.status_code, resp.json() if resp.text else {}
    except requests.exceptions.Timeout:
        return 0, {"error": "Request timed out (30s)"}
    except Exception as e:
        return 0, {"error": str(e)}


def api_get(path, token=None):
    """Make a GET request to the API."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{API_BASE}{path}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        return resp.status_code, resp.json() if resp.text else {}
    except Exception as e:
        return 0, {"error": str(e)}


# ============================================================
#  STEP 1: REGISTER TEST USERS
# ============================================================

def register_users(num_users):
    """Register test users via the auth API."""
    log(f"Registering {num_users} test users...")
    users = []

    for i in range(1, num_users + 1):
        username = f"{TEST_USER_PREFIX}{i}"
        email = f"{username}@{TEST_EMAIL_DOMAIN}"
        data = {
            "username": username,
            "email": email,
            "password": TEST_PASSWORD,
            "first_name": f"Load",
            "last_name": f"Tester{i}",
            "age": 25,
        }

        status, resp = api_post("/auth/api/auth/register", data)

        if status == 201:
            user_id = resp.get("user_id")
            users.append({"index": i, "username": username, "email": email, "user_id": user_id})
            log(f"  Registered {username} (ID: {user_id})")
        elif status == 422 and "already" in str(resp).lower():
            # User already exists from previous run
            log(f"  {username} already exists, will reuse")
            users.append({"index": i, "username": username, "email": email, "user_id": None})
        else:
            log(f"  FAILED to register {username}: {status} - {resp}")
            users.append({"index": i, "username": username, "email": email, "user_id": None, "error": True})

    log(f"Registration done: {len([u for u in users if not u.get('error')])} succeeded")
    return users


# ============================================================
#  STEP 2: VERIFY EMAILS IN DB
# ============================================================

def verify_emails(users):
    """Set email_verified_at directly in the auth DB."""
    log("Verifying emails in auth DB...")
    try:
        conn = mysql.connector.connect(**AUTH_DB_CONFIG)
        cursor = conn.cursor()

        for user in users:
            if user.get('error'):
                continue
            cursor.execute("""
                UPDATE users SET email_verified_at = NOW()
                WHERE email = %s AND email_verified_at IS NULL
            """, (user['email'],))

            # Also get user_id if we don't have it
            if user['user_id'] is None:
                cursor.execute("SELECT user_id FROM users WHERE email = %s", (user['email'],))
                row = cursor.fetchone()
                if row:
                    user['user_id'] = row[0]

        conn.commit()
        cursor.close()
        conn.close()
        verified = len([u for u in users if u.get('user_id')])
        log(f"Verified {verified} emails in DB")
    except Exception as e:
        log(f"ERROR verifying emails: {e}")
        sys.exit(1)


# ============================================================
#  STEP 3: LOGIN USERS
# ============================================================

def login_users(users):
    """Login each user to get Sanctum tokens."""
    log("Logging in users...")
    tokens = {}

    for user in users:
        if user.get('error') or not user.get('user_id'):
            continue

        data = {"email": user['email'], "password": TEST_PASSWORD}
        status, resp = api_post("/auth/api/auth/login", data)

        if status == 200 and resp.get('token'):
            tokens[user['user_id']] = resp['token']
            log(f"  Logged in {user['username']} (token obtained)")
        else:
            log(f"  FAILED to login {user['username']}: {status} - {json.dumps(resp)[:100]}")
            user['error'] = True

    log(f"Login done: {len(tokens)} tokens obtained")
    return tokens


# ============================================================
#  STEP 4: JOIN GROUP
# ============================================================

def join_group(users, tokens, group_id, group_code=None):
    """Have all users join the target group."""
    log(f"Joining {len(tokens)} users to group {group_id}...")
    success = 0

    for user in users:
        uid = user.get('user_id')
        if not uid or uid not in tokens:
            continue

        # Try join by code first (works for private groups), fallback to ID
        if group_code:
            status, resp = api_post(
                "/social/api/groups/join-with-code",
                data={"group_code": group_code},
                token=tokens[uid]
            )
        else:
            status, resp = api_post(f"/social/api/groups/{group_id}/join", token=tokens[uid])

        if status in (200, 201):
            log(f"  {user['username']} joined group {group_id}")
            success += 1
        elif "already" in str(resp).lower():
            log(f"  {user['username']} already in group")
            success += 1
        elif status == 400 and "full" in str(resp).lower():
            log(f"  GROUP IS FULL — cannot add more members")
            break
        else:
            log(f"  FAILED: {user['username']} - {status} - {json.dumps(resp)[:100]}")

    log(f"Group join done: {success} users in group")
    return success


# ============================================================
#  STEP 5: JOIN LOBBY (THE ACTUAL LOAD TEST)
# ============================================================

def join_lobby(users, tokens, session_id, stagger_ms=500):
    """Have all users join the lobby with timing measurements."""
    log(f"")
    log(f"{'='*60}")
    log(f"  LOBBY LOAD TEST: {len(tokens)} users joining {session_id}")
    log(f"  Stagger: {stagger_ms}ms between each user")
    log(f"{'='*60}")
    log(f"")

    results = []
    success = 0
    failed = 0

    for user in users:
        uid = user.get('user_id')
        if not uid or uid not in tokens:
            continue

        start = time.time()
        status, resp = api_post(f"/social/api/lobby/{session_id}/join", token=tokens[uid])
        elapsed = (time.time() - start) * 1000  # ms

        if status == 200:
            log(f"  [{success+1:>2d}] {user['username']:>20s} joined in {elapsed:>6.0f}ms")
            results.append({"user": user['username'], "status": "OK", "time_ms": elapsed})
            success += 1
        elif status == 409:
            msg = str(resp)
            if "already" in msg.lower() and "this lobby" in msg.lower():
                log(f"  [{success+1:>2d}] {user['username']:>20s} already in lobby")
                results.append({"user": user['username'], "status": "ALREADY_IN", "time_ms": elapsed})
                success += 1
            else:
                log(f"  [--] {user['username']:>20s} CONFLICT: {json.dumps(resp)[:80]}")
                results.append({"user": user['username'], "status": "CONFLICT", "time_ms": elapsed})
                failed += 1
        else:
            log(f"  [--] {user['username']:>20s} FAILED ({status}): {json.dumps(resp)[:80]}")
            results.append({"user": user['username'], "status": f"FAIL_{status}", "time_ms": elapsed})
            failed += 1

        # Stagger between joins
        if stagger_ms > 0:
            time.sleep(stagger_ms / 1000.0)

    # Report
    log(f"")
    log(f"{'='*60}")
    log(f"  RESULTS")
    log(f"{'='*60}")
    log(f"  Total users:     {success + failed}")
    log(f"  Successful:      {success}")
    log(f"  Failed:          {failed}")

    times = [r['time_ms'] for r in results if r['status'] == 'OK']
    if times:
        log(f"  Avg response:    {sum(times)/len(times):.0f}ms")
        log(f"  Min response:    {min(times):.0f}ms")
        log(f"  Max response:    {max(times):.0f}ms")
        slow = len([t for t in times if t > 2000])
        log(f"  Slow (>2s):      {slow}")
    log(f"{'='*60}")

    return results


# ============================================================
#  CLEANUP
# ============================================================

def cleanup():
    """Remove all loadtest users from the database."""
    log("Cleaning up load test users...")
    try:
        conn = mysql.connector.connect(**AUTH_DB_CONFIG)
        cursor = conn.cursor()

        # Get user IDs first
        cursor.execute(f"SELECT user_id FROM users WHERE username LIKE '{TEST_USER_PREFIX}%'")
        user_ids = [row[0] for row in cursor.fetchall()]

        if not user_ids:
            log("No load test users found")
            conn.close()
            return

        log(f"Found {len(user_ids)} load test users to remove")

        # Delete tokens
        ids_str = ','.join(str(uid) for uid in user_ids)
        cursor.execute(f"DELETE FROM personal_access_tokens WHERE tokenable_id IN ({ids_str})")
        log(f"  Deleted {cursor.rowcount} tokens")

        # Delete users
        cursor.execute(f"DELETE FROM users WHERE username LIKE '{TEST_USER_PREFIX}%'")
        log(f"  Deleted {cursor.rowcount} users")

        conn.commit()
        cursor.close()
        conn.close()
        log("Cleanup done")

        # Also clean from social DB
        log("Cleaning social DB...")
        social_config = AUTH_DB_CONFIG.copy()
        social_config['port'] = 3314
        social_config['database'] = 'fitnease_social_db'
        conn = mysql.connector.connect(**social_config)
        cursor = conn.cursor()

        ids_str = ','.join(str(uid) for uid in user_ids)
        cursor.execute(f"DELETE FROM group_members WHERE user_id IN ({ids_str})")
        log(f"  Removed {cursor.rowcount} group memberships")
        cursor.execute(f"DELETE FROM lobby_members WHERE user_id IN ({ids_str})")
        log(f"  Removed {cursor.rowcount} lobby memberships")

        conn.commit()
        cursor.close()
        conn.close()
        log("Social DB cleanup done")

    except Exception as e:
        log(f"ERROR during cleanup: {e}")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FitNEase Lobby Load Test')
    parser.add_argument('--group-id', type=int, help='Group ID to join')
    parser.add_argument('--group-code', type=str, help='Group code (for private groups)')
    parser.add_argument('--session-id', type=str, help='Lobby session ID to join')
    parser.add_argument('--num-users', type=int, default=24, help='Number of test users (default: 24)')
    parser.add_argument('--stagger', type=int, default=500, help='Milliseconds between each join (default: 500)')
    parser.add_argument('--cleanup', action='store_true', help='Remove all load test users')
    parser.add_argument('--skip-register', action='store_true', help='Skip registration (reuse existing users)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt before lobby join')
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    if not args.group_id or not args.session_id:
        print("Usage: python3 load_test_lobby.py --group-id <ID> --session-id <UUID> --num-users 24")
        print("       python3 load_test_lobby.py --group-id <ID> --group-code <CODE> --session-id <UUID>")
        print("       python3 load_test_lobby.py --cleanup")
        sys.exit(1)

    log(f"FitNEase Lobby Load Test")
    log(f"Target: {args.num_users} users → group {args.group_id} → lobby {args.session_id}")
    log(f"Stagger: {args.stagger}ms between joins")
    log(f"")

    # Step 1: Register
    if not args.skip_register:
        users = register_users(args.num_users)
    else:
        log("Skipping registration, reusing existing users...")
        users = []
        for i in range(1, args.num_users + 1):
            users.append({
                "index": i,
                "username": f"{TEST_USER_PREFIX}{i}",
                "email": f"{TEST_USER_PREFIX}{i}@{TEST_EMAIL_DOMAIN}",
                "user_id": None,
            })

    # Step 2: Verify emails
    verify_emails(users)

    # Step 3: Login
    tokens = login_users(users)

    if len(tokens) == 0:
        log("ERROR: No users could login. Aborting.")
        sys.exit(1)

    # Step 4: Join group
    join_group(users, tokens, args.group_id, group_code=args.group_code)

    # Step 5: Join lobby (load test)
    if not args.no_confirm:
        input_prompt = input("\nReady to start lobby load test? Press Enter (or 'q' to quit): ")
        if input_prompt.lower() == 'q':
            log("Aborted.")
            return

    results = join_lobby(users, tokens, args.session_id, stagger_ms=args.stagger)

    log(f"\nDone! Check your emulator to see {len([r for r in results if r['status'] in ('OK', 'ALREADY_IN')])} users in the lobby.")


if __name__ == '__main__':
    main()
