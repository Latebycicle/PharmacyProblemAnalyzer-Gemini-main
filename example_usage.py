#!/usr/bin/env python3
"""
Example usage script for Pharmacy Problem Analyzer

This script demonstrates how to use the APIs programmatically.
"""

import requests
import json
import os
from pathlib import Path

# Configuration
UPLOAD_URL = "http://localhost:5000/upload"
QUERY_URL = "http://localhost:5000/query"
HEALTH_URL = "http://localhost:5000/health"

def check_health():
    """Check if the API server is running."""
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            print("✅ API server is healthy")
            return True
        else:
            print(f"❌ API server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running?")
        return False

def upload_documents(file_paths):
    """Upload documents to the system."""
    print(f"📤 Uploading {len(file_paths)} documents...")
    
    files = []
    for file_path in file_paths:
        if Path(file_path).exists():
            files.append(('files', open(file_path, 'rb')))
        else:
            print(f"⚠️  File not found: {file_path}")
    
    if not files:
        print("❌ No valid files to upload")
        return False
    
    try:
        response = requests.post(UPLOAD_URL, files=files)
        
        # Close file handles
        for _, file_handle in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   Files processed: {result.get('files_processed', [])}")
            print(f"   Documents created: {result.get('documents_created', 0)}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def query_system(question):
    """Query the system with a question."""
    print(f"❓ Querying: {question}")
    
    try:
        response = requests.post(
            QUERY_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps({'query': question})
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful!")
            print(f"   Response: {result.get('response', 'No response')}")
            print(f"   Context length: {result.get('context_length', 0)} characters")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def main():
    """Main example workflow."""
    print("🏥 Pharmacy Problem Analyzer - Example Usage")
    print("=" * 50)
    
    # Check if server is running
    if not check_health():
        print("\n💡 To start the servers, run:")
        print("   python APIs/Uploadfilesapi.py  # In one terminal")
        print("   python APIs/QueryAPI.py        # In another terminal")
        return
    
    # Example 1: Upload sample documents
    print("\n📋 Step 1: Uploading sample documents")
    sample_files = [
        "sample_files/sample1.txt",
        "sample_files/sample2.txt",
        "sample_files/sample3.txt"
    ]
    
    if upload_documents(sample_files):
        print("✅ Documents uploaded successfully!")
    else:
        print("❌ Document upload failed")
        return
    
    # Example 2: Query the system
    print("\n📋 Step 2: Querying the system")
    
    example_questions = [
        "What are common issues with pharmacy billing systems?",
        "How can pharmacies improve medication adherence?",
        "What staffing problems do pharmacies typically face?",
        "What are the main causes of long wait times in pharmacies?"
    ]
    
    for question in example_questions:
        print(f"\n{'─' * 40}")
        query_system(question)
    
    print("\n✅ Example completed!")
    print("\n💡 You can now:")
    print("   - Run 'streamlit run ragtrial.py' for the web interface")
    print("   - Use the APIs directly with your own documents")
    print("   - Customize the system for your specific needs")

if __name__ == "__main__":
    main()