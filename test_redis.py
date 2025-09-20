# import redis
# import os
# from dotenv import load_dotenv

# # -------------------
# # Load environment variables
# # -------------------
# load_dotenv()
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_DB = int(os.getenv("REDIS_DB", 0))
# REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# # -------------------
# # Connect to Redis
# # -------------------
# r = redis.Redis(
#     host=REDIS_HOST,
#     port=REDIS_PORT,
#     db=REDIS_DB,
#     password=REDIS_PASSWORD,
#     decode_responses=True  # Keys/values as strings
# )

# # -------------------
# # List all keys
# # -------------------
# keys = r.keys("*")
# print(f"🔑 Keys in Redis DB {REDIS_DB}: {len(keys)} found")
# for key in keys:
#     print("-", key)

# # -------------------
# # Show values safely based on type
# # -------------------
# for key in keys:
#     key_type = r.type(key)
#     print(f"\nKey: {key} (type: {key_type})")

#     if key_type == "string":
#         print("Value:", r.get(key))
#     elif key_type == "hash":
#         print("Value:", r.hgetall(key))
#     elif key_type == "list":
#         print("Value:", r.lrange(key, 0, -1))
#     elif key_type == "set":
#         print("Value:", r.smembers(key))
#     elif key_type == "zset":
#         print("Value:", r.zrange(key, 0, -1, withscores=True))
#     else:
#         print("Value: <unknown type>")


import redis
import os
from dotenv import load_dotenv

# -------------------
# Load environment variables
# -------------------
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# -------------------
# Connect to Redis
# -------------------
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# -------------------
# Flush the database
# -------------------
r.flushdb()
print(f"✅ Redis DB {REDIS_DB} has been completely cleared.")
