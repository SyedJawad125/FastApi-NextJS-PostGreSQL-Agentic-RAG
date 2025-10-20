# from sqlalchemy.orm import Session
# from app.database import SessionLocal  # assuming SessionLocal returns a DB session
# from app.models.permission import Permission

# permissions = [
#     {"name": "Create Role", "code": "create_role", "module_name": "Role", "description": "User can create role"},
#     {"name": "Read Role", "code": "read_role", "module_name": "Role", "description": "User can read role"},
#     {"name": "Update Role", "code": "update_role", "module_name": "Role", "description": "User can update role"},
#     {"name": "Delete Role", "code": "delete_role", "module_name": "Role", "description": "User can delete role"},
    
#     {"name": "Create Employee", "code": "create_employee", "module_name": "Employee", "description": "User can create Employee"},
#     {"name": "Read Employee", "code": "read_employee", "module_name": "Employee", "description": "User can read Employee"},
#     {"name": "Update Employee", "code": "update_employee", "module_name": "Employee", "description": "User can update Employee"},
#     {"name": "Delete Employee", "code": "delete_employee", "module_name": "Employee", "description": "User can delete Employee"},

#     {"name": "Create Image", "code": "create_image", "module_name": "Image", "description": "User can create Image"},
#     {"name": "Read Image", "code": "read_image", "module_name": "Image", "description": "User can read Image"},
#     {"name": "Update Image", "code": "update_image", "module_name": "Image", "description": "User can update Image"},
#     {"name": "Delete Image", "code": "delete_image", "module_name": "Image", "description": "User can delete Image"},

#     {"name": "Create Image Category", "code": "create_image_category", "module_name": "Image Category", "description": "User can create Image Category"},
#     {"name": "Read Image Category", "code": "read_image_category", "module_name": "Image Category", "description": "User can read Image Category"},
#     {"name": "Update Image Category", "code": "update_image_category", "module_name": "Image Category", "description": "User can update Image Category"},
#     {"name": "Delete Image Category", "code": "delete_image_category", "module_name": "Image Category", "description": "User can delete Image Category"},
# ]

# def add_permissions_to_db(db: Session):
#     for perm in permissions:
#         existing = db.query(Permission).filter_by(code=perm["code"]).first()
#         if not existing:
#             new_perm = Permission(**perm)
#             db.add(new_perm)
#             print(f"✅ Added: {perm['name']}")
#         else:
#             print(f"⏩ Skipped (already exists): {perm['name']}")
#     db.commit()


# if __name__ == "__main__":
#     print("🚀 Populating permissions...")
#     db = SessionLocal()
#     try:
#         add_permissions_to_db(db)
#     finally:
#         db.close()




#!/usr/bin/env python3
"""
Test script to verify the RAG system works
"""
import asyncio
import sys
import os

# Add the app directory to path
sys.path.append(os.path.dirname(__file__))

async def test_system():
    print("🧪 Testing RAG System...")
    
    try:
        # Test dependencies
        from app.core.dependencies import get_llm_service, get_embedding_service, get_vectorstore
        from app.services.orchestrator import get_rag_orchestrator
        
        llm = get_llm_service()
        print("✅ LLM Service: OK")
        
        embedder = get_embedding_service()
        print("✅ Embedding Service: OK")
        
        vectorstore = get_vectorstore()
        print("✅ Vector Store: OK")
        
        # Test orchestrator
        rag_service = get_rag_orchestrator()
        print("✅ RAG Orchestrator: OK")
        
        # Test simple query
        print("\n🔍 Testing simple query...")
        result = await rag_service.execute_query("What is artificial intelligence?")
        print(f"✅ Query executed successfully!")
        print(f"📝 Answer: {result['answer'][:100]}...")
        print(f"⚡ Processing time: {result['processing_time']:.2f}s")
        print(f"🎯 Strategy: {result['strategy_used'].value}")
        print(f"📊 Retrieved chunks: {len(result['retrieved_chunks'])}")
        
        print("\n🎉 ALL TESTS PASSED! Your RAG system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)