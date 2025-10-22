from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import json
from openai import OpenAI
from pymongo import MongoClient
from serpapi import GoogleSearch
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

class PepperRAG:
    def __init__(self, openai_api_key, mongodb_uri, db_name, collection_name):
        self.client = OpenAI(api_key=openai_api_key)
        try:
            self.mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.mongo_client.server_info()
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
            print(f"✅ MongoDB connected successfully to {db_name}.{collection_name}")
        except Exception as e:
            print(f"❌ MongoDB connection error: {str(e)}")
            raise
        self.model = "gpt-4o-mini"
        
    def _call_llm(self, system_prompt, user_prompt, response_format="text"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if response_format == "json":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
        
        return response.choices[0].message.content
    
    def step1_check_relevance(self, question):
        system_prompt = """You are a relevance checker. Determine if the user's question is related to peppers, chili, or cabai (Indonesian for chili).
Return a JSON object with:
- "relevant": true or false
- "reason": brief explanation"""
        
        user_prompt = f"Question: {question}"
        response = self._call_llm(system_prompt, user_prompt, response_format="json")
        result = json.loads(response)
        return result['relevant'], result['reason']
    
    def step2_check_document_capability(self, question, schema_info):
        system_prompt = f"""You are a document capability analyzer. Given the MongoDB schema/properties and a question, determine if the document store can answer it.

MongoDB Collection Schema/Properties:
{schema_info}

Return a JSON object with:
- "can_answer": true or false
- "reasoning": brief explanation of why the documents can or cannot answer the question

IMPORTANT: Be generous in determining if the database can answer. If the schema has relevant fields, return true."""
        
        user_prompt = f"Question: {question}"
        response = self._call_llm(system_prompt, user_prompt, response_format="json")
        result = json.loads(response)
        print(f"📊 Step 2 Result - Can Answer: {result['can_answer']}, Reason: {result['reasoning']}")
        return result['can_answer'], result['reasoning']
    
    def step3_generate_mongodb_query(self, question, schema_info):
        system_prompt = f"""You are a MongoDB query generator. Generate a MongoDB find query based on the user's question.

MongoDB Collection Schema/Properties:
{schema_info}

Return a JSON object with:
- "query": MongoDB query object (can be empty {{}} for find all)
- "projection": projection object (optional, can be null)
- "sort": sort object (optional, can be null)
- "limit": number (optional, can be null)

IMPORTANT: 
- For min/max queries, use proper MongoDB operators ($gte, $lte, etc)
- For name searches, use $regex with case-insensitive option
- Some names don't include 'cabai', be flexible with search terms"""
        
        user_prompt = f"Question: {question}"
        response = self._call_llm(system_prompt, user_prompt, response_format="json")
        query_info = json.loads(response)
        print(f"🔍 Generated Query: {json.dumps(query_info, indent=2)}")
        return query_info
    
    def step4_execute_query(self, query_info):
        try:
            query = query_info.get('query', {})
            projection = query_info.get('projection')
            sort = query_info.get('sort')
            limit = query_info.get('limit')
            
            cursor = self.collection.find(query, projection)
            
            if sort:
                cursor = cursor.sort(list(sort.items()))
            if limit:
                cursor = cursor.limit(limit)
            
            results = list(cursor)
            print(f"📦 Query executed: Found {len(results)} documents")
            return results
        except Exception as e:
            print(f"❌ Query execution error: {str(e)}")
            traceback.print_exc()
            return []
    
    def step5_evaluate_results(self, question, results):
        system_prompt = """You are a result evaluator. Determine if the MongoDB results adequately answer the user's question.

Return a JSON object with:
- "answers_question": true or false
- "reasoning": brief explanation

IMPORTANT: If there are relevant results, be generous and return true."""
        
        results_str = json.dumps(results, indent=2, default=str)[:2000]
        user_prompt = f"""Question: {question}

MongoDB Results:
{results_str}"""
        
        response = self._call_llm(system_prompt, user_prompt, response_format="json")
        result = json.loads(response)
        print(f"✅ Step 5 Result - Answers Question: {result['answers_question']}, Reason: {result['reasoning']}")
        return result['answers_question'], result['reasoning']
    
    def step6_web_search(self, question):
        api_key = os.getenv('SERPAPI_KEY', '7f4aebfaeec551100c8e71e0f8a7f8ca2a7562943a8943e52adea01a3b7383da')
        
        search = GoogleSearch({
            "q": question,
            "engine": "google",
            "api_key": api_key,
            "num": 5
        })
        
        results = search.get_dict()
        
        web_results = []
        for r in results.get("organic_results", []):
            web_results.append({
                "title": r.get("title"),
                "snippet": r.get("snippet"),
                "url": r.get("link")
            })
        
        print(f"🌐 Web search completed: Found {len(web_results)} results")
        return web_results
    
    def step7_generate_final_answer(self, question, mongodb_results, web_results=None):
        system_prompt = """You are a helpful assistant specializing in peppers and chili. 
Generate a natural, conversational answer to the user's question based on the provided information.
Be concise but informative. If the information is insufficient, say so honestly."""
        
        context = f"Question: {question}\n\n"
        
        if mongodb_results:
            context += "Information from database:\n"
            context += json.dumps(mongodb_results, indent=2, default=str)[:3000]
            context += "\n\n"
        
        if web_results:
            context += "Additional information from web:\n"
            context += json.dumps(web_results, indent=2)[:1000]
        
        return self._call_llm(system_prompt, context)
    
    def get_schema_info(self):
        """
        Get schema information from MongoDB collection.
        Includes schema and sample values of 'Nama' field (if available).
        """
        # Get a sample document to infer schema
        sample = self.collection.find_one()

        if sample:
            # Remove _id for cleaner schema representation
            if '_id' in sample:
                del sample['_id']

            # Create schema description
            schema_info = "Available properties:\n"
            for key in sample.keys():
                schema_info += f"- {key}: {type(sample[key]).__name__}\n"

            # Get all unique Nama values (if the field exists)
            nama_values = self.collection.distinct("Nama")
            if nama_values:
                schema_info += "\nSample 'Nama' values:\n"
                for n in nama_values:
                    schema_info += f"- {n}\n"
            else:
                schema_info += "\nNo 'Nama' field found in documents.\n"

            return schema_info
        else:
            return "No documents found in collection. Please add your schema information manually."

    
# Initialize RAG system
print("🚀 Initializing PepperRAG...")
try:
    rag = PepperRAG(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        mongodb_uri=os.getenv('MONGODB_URI'),
        db_name=os.getenv('DB_NAME', 'cabai_rag'),
        collection_name=os.getenv('COLLECTION_NAME', 'cabai')
    )
    print("✅ PepperRAG initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize PepperRAG: {str(e)}")
    rag = None

@app.route('/api/chat', methods=['POST'])
def chat():
    if not rag:
        return jsonify({"error": "RAG system not initialized. Check MongoDB connection."}), 500
    
    data = request.json
    question = data.get('question', '')
    
    print(f"\n{'='*50}")
    print(f"📨 New Question: {question}")
    print(f"{'='*50}\n")
    
    def generate():
        try:
            # Step 1: Check relevance
            yield f"data: {json.dumps({'step': 1, 'status': 'processing', 'message': 'Checking question relevance...'})}\n\n"
            
            is_relevant, reason = rag.step1_check_relevance(question)
            print(f"✅ Step 1 - Relevant: {is_relevant}, Reason: {reason}")
            yield f"data: {json.dumps({'step': 1, 'status': 'complete', 'message': f'Relevance: {reason}'})}\n\n"
            
            if not is_relevant:
                yield f"data: {json.dumps({'step': 'final', 'answer': 'I apologize, but your question does not appear to be related to peppers or chili. Please ask me about peppers, chili, or cabai!', 'source': 'relevance_check'})}\n\n"
                return
            
            # Get schema
            schema_info = rag.get_schema_info()
            
            # Step 2: Check document capability
            yield f"data: {json.dumps({'step': 2, 'status': 'processing', 'message': 'Analyzing database capabilities...'})}\n\n"
            
            can_answer, reasoning = rag.step2_check_document_capability(question, schema_info)
            yield f"data: {json.dumps({'step': 2, 'status': 'complete', 'message': f'Database check: {reasoning}'})}\n\n"
            
            mongodb_results = []
            
            if can_answer:
                # Step 3: Generate query
                yield f"data: {json.dumps({'step': 3, 'status': 'processing', 'message': 'Generating database query...'})}\n\n"
                
                query_info = rag.step3_generate_mongodb_query(question, schema_info)
                yield f"data: {json.dumps({'step': 3, 'status': 'complete', 'message': 'Query generated successfully'})}\n\n"
                
                # Step 4: Execute query
                yield f"data: {json.dumps({'step': 4, 'status': 'processing', 'message': 'Executing database query...'})}\n\n"
                
                mongodb_results = rag.step4_execute_query(query_info)
                yield f"data: {json.dumps({'step': 4, 'status': 'complete', 'message': f'Found {len(mongodb_results)} results'})}\n\n"
                
                # Step 5: Evaluate results
                yield f"data: {json.dumps({'step': 5, 'status': 'processing', 'message': 'Evaluating results...'})}\n\n"
                
                answers_question, eval_reasoning = rag.step5_evaluate_results(question, mongodb_results)
                yield f"data: {json.dumps({'step': 5, 'status': 'complete', 'message': f'Evaluation: {eval_reasoning}'})}\n\n"
                
                if answers_question:
                    # Step 7: Generate answer
                    yield f"data: {json.dumps({'step': 7, 'status': 'processing', 'message': 'Generating answer...'})}\n\n"
                    
                    answer = rag.step7_generate_final_answer(question, mongodb_results)
                    
                    # Extract images from results
                    images = []
                    for doc in mongodb_results:
                        if 'gambar' in doc and doc['gambar']:
                            images.append(doc['gambar'])
                    
                    print(f"✅ Final answer generated from MongoDB only")
                    yield f"data: {json.dumps({'step': 'final', 'answer': answer, 'source': 'mongodb_only', 'images': images})}\n\n"
                    return
            
            # Step 6: Web search
            yield f"data: {json.dumps({'step': 6, 'status': 'processing', 'message': 'Searching the web...'})}\n\n"
            
            web_results = rag.step6_web_search(question)
            yield f"data: {json.dumps({'step': 6, 'status': 'complete', 'message': f'Found {len(web_results)} web results'})}\n\n"
            
            # Step 7: Generate final answer
            yield f"data: {json.dumps({'step': 7, 'status': 'processing', 'message': 'Generating answer...'})}\n\n"
            
            answer = rag.step7_generate_final_answer(question, mongodb_results, web_results)
            
            # Extract images
            images = []
            for doc in mongodb_results:
                if 'gambar' in doc and doc['gambar']:
                    images.append(doc['gambar'])
            
            source = 'mongodb_and_web' if mongodb_results else 'web_only'
            print(f"✅ Final answer generated from {source}")
            yield f"data: {json.dumps({'step': 'final', 'answer': answer, 'source': source, 'images': images})}\n\n"
        
        except Exception as e:
            print(f"❌ Error in chat endpoint: {str(e)}")
            traceback.print_exc()
            yield f"data: {json.dumps({'step': 'final', 'answer': f'An error occurred: {str(e)}', 'source': 'error'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health():
    mongo_status = "connected" if rag and rag.mongo_client else "disconnected"
    return jsonify({
        "status": "ok",
        "mongodb": mongo_status,
        "db": os.getenv('DB_NAME', 'cabai_rag'),
        "collection": os.getenv('COLLECTION_NAME', 'cabai')
    })

@app.route('/api/test-db', methods=['GET'])
def test_db():
    """Test endpoint to verify MongoDB connection and data"""
    if not rag:
        return jsonify({"error": "RAG not initialized"}), 500
    
    try:
        count = rag.collection.count_documents({})
        sample = rag.collection.find_one()
        schema = rag.get_schema_info()
        
        return jsonify({
            "status": "ok",
            "document_count": count,
            "sample_document": json.loads(json.dumps(sample, default=str)),
            "schema": schema
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))