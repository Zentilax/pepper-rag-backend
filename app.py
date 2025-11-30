from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import json
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import traceback
import requests

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)

# FIXED CORS CONFIGURATION
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

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
        self.doc_api_url = "https://doc-pepper-rag-backend-production.up.railway.app/query"
        
    def _call_llm(self, system_prompt, user_prompt, response_format="text", use_cheap_model=False):
        if use_cheap_model:
            # GPT-5-nano has different calling convention
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": combined_prompt}],
            )
            content = dict(response.choices[0].message)["content"].strip()
            return content
        else:
            # Standard model calling convention
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
    
    def step5a_query_document_api(self, question, top_k=2):
        """Query the document API with the user's question"""
        try:
            print(f"📄 Querying document API with: {question}")
            response = requests.post(
                self.doc_api_url,
                json={
                    "question": question,
                    "top_k": top_k
                },
                timeout=30
            )
            response.raise_for_status()
            doc_results = response.json()
            print(f"📄 Document API returned: {len(doc_results.get('results', []))} results")
            return doc_results
        except Exception as e:
            print(f"❌ Document API error: {str(e)}")
            traceback.print_exc()
            return None
    
    def step5b_evaluate_document_results(self, question, doc_results):
        """Evaluate if document API results can answer the question using cheap model"""
        system_prompt = """You are a result evaluator. Determine if the document retrieval results adequately answer the user's question.

        Return a JSON object with:
        - "answers_question": true or false
        - "reasoning": brief explanation

        IMPORTANT: If there are relevant results that provide a good answer, return true."""
        
        doc_str = json.dumps(doc_results, indent=2, default=str)[:2000]
        user_prompt = f"""Question: {question}

        Document Retrieval Results:
        {doc_str}"""
        
        response = self._call_llm(system_prompt, user_prompt, response_format="json", use_cheap_model=True)
        result = json.loads(response)
        print(f"✅ Step 5b Result - Answers Question: {result['answers_question']}, Reason: {result['reasoning']}")
        return result['answers_question'], result['reasoning']
    
    def step6_web_search(self, question):
        """Perform web search using OpenAI's Web Search tool"""
        try:
            print(f"🌐 [WEB_SEARCH] Starting web search for: {question}", flush=True)
            
            response = self.client.responses.create(
                model="gpt-4.1",
                input=question,
                tools=[{"type": "web_search"}]
            )
            
            web_results = []
            
            # Debug: Print raw response structure
            print(f"🔍 [WEB_SEARCH] Raw response type: {type(response)}", flush=True)
            print(f"🔍 [WEB_SEARCH] Raw response output type: {type(response.output)}", flush=True)
            print(f"🔍 [WEB_SEARCH] Raw response output length: {len(response.output) if hasattr(response.output, '__len__') else 'N/A'}", flush=True)
            
            for idx, item in enumerate(response.output):
                print(f"🔍 [WEB_SEARCH] Processing output item {idx}, type: {type(item)}", flush=True)
                if hasattr(item, "content"):
                    print(f"🔍 [WEB_SEARCH] Item {idx} has content, length: {len(item.content) if hasattr(item.content, '__len__') else 'N/A'}", flush=True)
                    for part_idx, part in enumerate(item.content):
                        print(f"🔍 [WEB_SEARCH] Content part {part_idx} type: {part.type}", flush=True)
                        if part.type == "web_search.results":
                            print(f"🔍 [WEB_SEARCH] Found web_search.results, processing {len(part.results)} results", flush=True)
                            for r_idx, r in enumerate(part.results):
                                result = {
                                    "title": r.get("title", "Untitled"),
                                    "snippet": r.get("snippet", ""),
                                    "url": r.get("url", "")
                                }
                                print(f"🔍 [WEB_SEARCH] Result {r_idx}: {json.dumps(result, indent=2)}", flush=True)
                                web_results.append(result)
                else:
                    print(f"🔍 [WEB_SEARCH] Item {idx} has no content attribute", flush=True)
            
            print(f"🌐 [WEB_SEARCH] COMPLETED: Found {len(web_results)} results", flush=True)
            print(f"🌐 [WEB_SEARCH] Final web_results: {json.dumps(web_results, indent=2)}", flush=True)
            return web_results
        except Exception as e:
            print(f"❌ [WEB_SEARCH] ERROR: {str(e)}", flush=True)
            traceback.print_exc()
            return []

    def _format_references(self, mongodb_results, doc_results, web_results):
        """
        Format references section for the final answer
        Returns: str
        """
        print(f"📚 [REFERENCES] Starting reference formatting", flush=True)
        print(f"📚 [REFERENCES] MongoDB results: {len(mongodb_results) if mongodb_results else 0}", flush=True)
        print(f"📚 [REFERENCES] Doc results: {len(doc_results.get('results', [])) if doc_results else 0}", flush=True)
        print(f"📚 [REFERENCES] Web results: {len(web_results) if web_results else 0}", flush=True)
        
        references = "\n\n---\n**References:**\n"
        ref_count = 0
        
        # Add MongoDB references
        if mongodb_results and len(mongodb_results) > 0:
            print(f"📚 [REFERENCES] Adding {len(mongodb_results)} MongoDB references", flush=True)
            references += "\n*From Database:*\n"
            for idx, doc in enumerate(mongodb_results, 1):
                ref_count += 1
                doc_name = doc.get('Nama', doc.get('name', doc.get('nama', f'Document {idx}')))
                references += f"[{ref_count}] {doc_name} (Internal Database)\n"
        
        # Add Document API references
        if doc_results and doc_results.get('results') and len(doc_results['results']) > 0:
            print(f"📚 [REFERENCES] Adding {len(doc_results['results'])} document references", flush=True)
            references += "\n*From Document Retrieval System:*\n"
            for idx, result in enumerate(doc_results['results'], 1):
                ref_count += 1
                doc_text = result.get('text', '')[:100]  # First 100 chars
                score = result.get('score', 0)
                references += f"[{ref_count}] Document (similarity: {score:.3f})\n    Preview: {doc_text}...\n"
        
        # Add Web references - FIXED
        if web_results and len(web_results) > 0:
            print(f"📚 [REFERENCES] Adding {len(web_results)} web references", flush=True)
            references += "\n*From Web Search:*\n"
            for idx, result in enumerate(web_results, 1):
                ref_count += 1
                title = result.get('title', 'Web Result')
                url = result.get('url', 'N/A')
                snippet = result.get('snippet', '')
                
                print(f"📚 [REFERENCES] Web ref {ref_count}: title={title}, url={url[:50] if url else 'N/A'}", flush=True)
                
                references += f"[{ref_count}] {title}\n"
                if url and url != 'N/A':
                    references += f"    URL: {url}\n"
                if snippet:
                    references += f"    Preview: {snippet[:100]}...\n"
        else:
            print(f"📚 [REFERENCES] No web results to add (web_results={web_results})", flush=True)
        
        if ref_count == 0:
            print(f"📚 [REFERENCES] No references found", flush=True)
            references += "No external references used.\n"
        
        print(f"📚 [REFERENCES] COMPLETED: Formatted {ref_count} total references", flush=True)
        return references

    def step7_generate_final_answer(self, question, mongodb_results=None, doc_results=None, web_results=None):
        """
        Generate humanized final answer with references
        Returns: str
        """
        print(f"🤖 [GENERATE_ANSWER] Starting answer generation", flush=True)
        print(f"🤖 [GENERATE_ANSWER] Input - MongoDB: {len(mongodb_results) if mongodb_results else 0}, Docs: {bool(doc_results)}, Web: {len(web_results) if web_results else 0}", flush=True)
        
        system_prompt = """You are a helpful assistant specializing in peppers and chili. 
    Generate a natural, conversational answer to the user's question based on the provided information.
    Be concise but informative. If the information is insufficient, say so honestly.

    IMPORTANT: When citing information, use inline citations like [1], [2], etc. to reference the sources.
    Number the sources based on the order they appear in the context (database results first, then document results, then web results)."""
        
        context = f"Question: {question}\n\n"
        
        source_count = 0
        
        # Add MongoDB results
        if mongodb_results and len(mongodb_results) > 0:
            print(f"🤖 [GENERATE_ANSWER] Adding {len(mongodb_results)} MongoDB results to context", flush=True)
            context += "Information from database:\n"
            for idx, doc in enumerate(mongodb_results, 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(doc, indent=2, default=str)}\n"
            context += "\n"
        
        # Add Document API results
        if doc_results and doc_results.get('results') and len(doc_results['results']) > 0:
            print(f"🤖 [GENERATE_ANSWER] Adding {len(doc_results['results'])} document results to context", flush=True)
            context += "Information from document retrieval:\n"
            for idx, result in enumerate(doc_results['results'], 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(result, indent=2, default=str)}\n"
            context += "\n"
        
        # Add Web results - FIXED
        if web_results and len(web_results) > 0:
            print(f"🤖 [GENERATE_ANSWER] Adding {len(web_results)} web results to context", flush=True)
            context += "Additional information from web:\n"
            for idx, result in enumerate(web_results, 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(result, indent=2)}\n"
            context += "\n"
        else:
            print(f"🤖 [GENERATE_ANSWER] No web results to add (web_results={web_results})", flush=True)
        
        # Add instruction to cite sources
        context += "\nRemember to cite sources using [1], [2], etc. in your answer."
        
        print(f"🤖 [GENERATE_ANSWER] Calling LLM with {source_count} sources", flush=True)
        answer = self._call_llm(system_prompt, context)
        
        print(f"🤖 [GENERATE_ANSWER] LLM response received, length: {len(answer)}", flush=True)
        
        # Add formatted references at the end
        references = self._format_references(mongodb_results, doc_results, web_results)
        final_answer = answer + references
        
        print(f"✅ [GENERATE_ANSWER] COMPLETED: Final answer length: {len(final_answer)}", flush=True)
        return final_answer
        
    
    def step7_generate_final_answer_old(self, question, mongodb_results=None, doc_results=None, web_results=None):
        """
        Generate humanized final answer with references
        Returns: str
        """
        system_prompt = """You are a helpful assistant specializing in peppers and chili. 
Generate a natural, conversational answer to the user's question based on the provided information.
Be concise but informative. If the information is insufficient, say so honestly.

IMPORTANT: When citing information, use inline citations like [1], [2], etc. to reference the sources.
Number the sources based on the order they appear in the context (database results first, then document results, then web results)."""
        
        context = f"Question: {question}\n\n"
        
        source_count = 0
        
        # Add MongoDB results
        if mongodb_results:
            context += "Information from database:\n"
            for idx, doc in enumerate(mongodb_results, 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(doc, indent=2, default=str)}\n"
            context += "\n"
        
        # Add Document API results
        if doc_results and doc_results.get('results'):
            context += "Information from document retrieval:\n"
            for idx, result in enumerate(doc_results['results'], 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(result, indent=2, default=str)}\n"
            context += "\n"
        
        # Add Web results
        if web_results:
            context += "Additional information from web:\n"
            for idx, result in enumerate(web_results, 1):
                source_count += 1
                context += f"[{source_count}] {json.dumps(result, indent=2)}\n"
        
        # Add instruction to cite sources
        context += "\nRemember to cite sources using [1], [2], etc. in your answer."
        
        answer = self._call_llm(system_prompt, context)
        
        # Add formatted references at the end
        references = self._format_references(mongodb_results, doc_results, web_results)
        final_answer = answer + references
        
        print(f"✅ Final answer generated with references")
        return final_answer
    
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
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    if not rag:
        return jsonify({"error": "RAG system not initialized. Check MongoDB connection."}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {str(e)}"}), 400
    
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
                no_ref_answer = "I apologize, but your question does not appear to be related to peppers or chili. Please ask me about peppers, chili, or cabai!\n\n---\n**References:**\nNo references available."
                yield f"data: {json.dumps({'step': 'final', 'answer': no_ref_answer, 'source': 'relevance_check'})}\n\n"
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
                    
                    answer = rag.step7_generate_final_answer(question, mongodb_results=mongodb_results)
                    
                    # Extract images from results
                    images = []
                    for doc in mongodb_results:
                        if 'gambar' in doc and doc['gambar']:
                            images.append(doc['gambar'])
                    
                    print(f"✅ Final answer generated from MongoDB only")
                    yield f"data: {json.dumps({'step': 'final', 'answer': answer, 'source': 'mongodb_only', 'images': images})}\n\n"
                    return
            
            # Step 5a: Query Document API
            yield f"data: {json.dumps({'step': '5a', 'status': 'processing', 'message': 'Querying document retrieval system...'})}\n\n"
            
            doc_results = rag.step5a_query_document_api(question)
            
            if doc_results:
                yield f"data: {json.dumps({'step': '5a', 'status': 'complete', 'message': 'Document retrieval complete'})}\n\n"
                
                # Step 5b: Evaluate document results using cheap model
                yield f"data: {json.dumps({'step': '5b', 'status': 'processing', 'message': 'Evaluating document results...'})}\n\n"
                
                doc_answers, doc_eval_reasoning = rag.step5b_evaluate_document_results(question, doc_results)
                yield f"data: {json.dumps({'step': '5b', 'status': 'complete', 'message': f'Document evaluation: {doc_eval_reasoning}'})}\n\n"
                
                if doc_answers:
                    # Step 7: Generate answer from documents
                    yield f"data: {json.dumps({'step': 7, 'status': 'processing', 'message': 'Generating answer from documents...'})}\n\n"
                    
                    answer = rag.step7_generate_final_answer(question, mongodb_results=mongodb_results, doc_results=doc_results)
                    
                    # Extract images
                    images = []
                    for doc in mongodb_results:
                        if 'gambar' in doc and doc['gambar']:
                            images.append(doc['gambar'])
                    
                    source = 'mongodb_and_documents' if mongodb_results else 'documents_only'
                    print(f"✅ Final answer generated from {source}")
                    yield f"data: {json.dumps({'step': 'final', 'answer': answer, 'source': source, 'images': images})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'step': '5a', 'status': 'complete', 'message': 'Document retrieval unavailable'})}\n\n"
            
            # Step 6: Web search
            yield f"data: {json.dumps({'step': 6, 'status': 'processing', 'message': 'Searching the web...'})}\n\n"
            
            web_results = rag.step6_web_search(question)
            yield f"data: {json.dumps({'step': 6, 'status': 'complete', 'message': f'Found {len(web_results)} web results'})}\n\n"
            
            # Step 7: Generate final answer
            yield f"data: {json.dumps({'step': 7, 'status': 'processing', 'message': 'Generating answer...'})}\n\n"
            
            answer = rag.step7_generate_final_answer(question, mongodb_results=mongodb_results, doc_results=doc_results, web_results=web_results)
            
            # Extract images
            images = []
            for doc in mongodb_results:
                if 'gambar' in doc and doc['gambar']:
                    images.append(doc['gambar'])
            
            # Determine source
            if mongodb_results and doc_results:
                source = 'mongodb_documents_and_web'
            elif mongodb_results:
                source = 'mongodb_and_web'
            elif doc_results:
                source = 'documents_and_web'
            else:
                source = 'web_only'
            
            print(f"✅ Final answer generated from {source}")
            yield f"data: {json.dumps({'step': 'final', 'answer': answer, 'source': source, 'images': images})}\n\n"
        
        except Exception as e:
            print(f"❌ Error in chat endpoint: {str(e)}")
            traceback.print_exc()
            error_answer = f"An error occurred: {str(e)}\n\n---\n**References:**\nNo references available."
            yield f"data: {json.dumps({'step': 'final', 'answer': error_answer, 'source': 'error'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    
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
    if request.method == 'OPTIONS':
        return '', 204
    
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