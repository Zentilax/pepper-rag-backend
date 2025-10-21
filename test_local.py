"""
Test script to verify your local setup is working correctly
Run this before starting the Flask app
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI

# Load environment variables
load_dotenv()

print("🧪 Testing PepperRAG Local Setup\n")
print("=" * 60)

# Test 1: Environment Variables
print("\n1️⃣  Testing Environment Variables...")
tests_passed = 0
tests_failed = 0

env_vars = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MONGODB_URI': os.getenv('MONGODB_URI','mongodb://localhost:27017/'),
    'DB_NAME': os.getenv('DB_NAME', 'cabai_rag'),
    'COLLECTION_NAME': os.getenv('COLLECTION_NAME', 'cabai'),
    'SERPAPI_KEY': os.getenv('SERPAPI_KEY')
}

for var_name, var_value in env_vars.items():
    if var_value:
        # Show partial value for security
        if 'KEY' in var_name:
            display_value = f"{var_value[:15]}..." if len(var_value) > 15 else "***"
        else:
            display_value = var_value
        print(f"   ✅ {var_name}: {display_value}")
        tests_passed += 1
    else:
        print(f"   ❌ {var_name}: NOT SET")
        tests_failed += 1

# Test 2: MongoDB Connection
print("\n2️⃣  Testing MongoDB Connection...")
try:
    client = MongoClient(env_vars['MONGODB_URI'], serverSelectionTimeoutMS=5000)
    # Try to get server info
    client.server_info()
    db = client[env_vars['DB_NAME']]
    collection = db[env_vars['COLLECTION_NAME']]
    
    doc_count = collection.count_documents({})
    print(f"   ✅ Connected to MongoDB")
    print(f"   ✅ Database: {env_vars['DB_NAME']}")
    print(f"   ✅ Collection: {env_vars['COLLECTION_NAME']}")
    print(f"   ✅ Documents: {doc_count}")
    
    if doc_count == 0:
        print(f"   ⚠️  Warning: Collection is empty. Run 'python seed_data.py' to add sample data")
    
    tests_passed += 1
    client.close()
    
except Exception as e:
    print(f"   ❌ MongoDB connection failed: {e}")
    print(f"   💡 Make sure MongoDB is running or check your MONGODB_URI")
    tests_failed += 1

# Test 3: OpenAI API
print("\n3️⃣  Testing OpenAI API...")
try:
    client = OpenAI(api_key=env_vars['OPENAI_API_KEY'])
    # Simple test request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API works!'"}],
        max_tokens=10
    )
    
    if response.choices[0].message.content:
        print(f"   ✅ OpenAI API is working")
        print(f"   ✅ Model: gpt-4o-mini")
        tests_passed += 1
    
except Exception as e:
    print(f"   ❌ OpenAI API test failed: {e}")
    print(f"   💡 Check your OPENAI_API_KEY is valid")
    tests_failed += 1

# Test 4: Port Availability
print("\n4️⃣  Testing Port Availability...")
import socket

def is_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

port = int(os.getenv('PORT', 5000))
if is_port_available(port):
    print(f"   ✅ Port {port} is available")
    tests_passed += 1
else:
    print(f"   ⚠️  Port {port} is already in use")
    print(f"   💡 Stop any running Flask apps or change PORT in .env")
    tests_failed += 1

# Test 5: Python Packages
print("\n5️⃣  Testing Python Packages...")
required_packages = [
    'flask',
    'flask_cors',
    'openai',
    'pymongo',
    'serpapi',
    'dotenv'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
        tests_passed += 1
    except ImportError:
        print(f"   ❌ {package} not installed")
        print(f"   💡 Run: pip install -r requirements.txt")
        tests_failed += 1

# Summary
print("\n" + "=" * 60)
print(f"📊 Test Summary:")
print(f"   ✅ Passed: {tests_passed}")
print(f"   ❌ Failed: {tests_failed}")

if tests_failed == 0:
    print(f"\n🎉 All tests passed! You're ready to run the app!")
    print(f"\n▶️  Next steps:")
    print(f"   1. If collection is empty: python seed_data.py")
    print(f"   2. Start backend: python app.py")
    print(f"   3. Open frontend: open index.html")
    print(f"   4. Configure frontend: Settings → http://localhost:{port}")
else:
    print(f"\n⚠️  Some tests failed. Please fix the issues above before running the app.")
    
    if 'OPENAI_API_KEY' not in os.environ or not os.getenv('OPENAI_API_KEY'):
        print(f"\n💡 Quick fix for missing .env:")
        print(f"   Create a .env file in the backend folder with:")
        print(f"   OPENAI_API_KEY=your-key-here")
        print(f"   MONGODB_URI=mongodb://localhost:27017/")
        print(f"   SERPAPI_KEY=your-key-here")

print("\n" + "=" * 60)