import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain imports (v0.3.x compatible)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Additional imports
import warnings
warnings.filterwarnings('ignore')
import getpass
import os
import chromadb

# ----------------------------------------------------------------------
# Prompt user for OpenAI API key securely
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ----------------------------------------------------------------------
chroma_client = chromadb.CloudClient(
  api_key='ck-4B9mg6mzjdkBmh78pxKaUyCDci9DdfUUrAEbPkMXee2Z',
  tenant='43fde602-a9c4-46e5-9947-6cad654de3ca',
  database='policy_details'
)

collection = chroma_client.get_collection(name="policy_embeddings")

vectorstore = Chroma(
    client=chroma_client,
    collection_name="policy_embeddings",
    embedding_function=embeddings
)
# ----------------------------------------------------------------------
#              ENHANCED JARGON SIMPLIFIER

class EnhancedJargonSimplifier:
    """Simplifies policy language while maintaining accuracy and citations."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        
    def simplify_with_citations(self, policy_text: str, user_query: str, documents: List[Document]) -> str:
        """Simplify policy text while maintaining accuracy and adding citations."""
        
        # Get policy names for citations
        policy_names = []
        for doc in documents:
            policy_name = doc.metadata.get('policy_type', 'Unknown Policy')
            if policy_name not in policy_names:
                policy_names.append(policy_name)
        
        policy_context = ", ".join(policy_names)
        
        simplification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at simplifying complex HR policy language while maintaining complete accuracy.

            Your goals:
            1. Make complex policy text easy to understand
            2. Maintain 100% accuracy - do not change any facts, numbers, or requirements
            3. Include proper citations to specific policies
            4. Use simple, clear language
            5. Organize information with bullet points when helpful
            6. Explain technical terms in parentheses
            7. Keep all specific details like timeframes, amounts, and eligibility requirements

            Guidelines:
            - Use "you" instead of "the employee"
            - Replace jargon with simple terms but keep the meaning exact
            - Add citations like "According to the [Policy Name]..."
            - If there are conditions or exceptions, state them clearly
            - Use bullet points for lists of benefits or requirements
            - Maintain professional but friendly tone"""),
            
            ("user", """User Question: {user_query}

            Policy Information to Simplify:
            {policy_text}

            Available Policy Documents: {policy_context}

            Please simplify this information while maintaining complete accuracy and including proper citations.""")
        ])
        
        simplification_chain = simplification_prompt | self.llm | StrOutputParser()
        
        simplified = simplification_chain.invoke({
            "user_query": user_query,
            "policy_text": policy_text,
            "policy_context": policy_context
        })
        
        return simplified

# Initialize enhanced simplifier
enhanced_simplifier = EnhancedJargonSimplifier()

# ----------------------------------------------------------------------
#                    IMPROVED POLICY RETRIEVER
class ImprovedPolicyRetriever:
    """Enhanced retrieval system with better document matching and filtering."""
    
    def __init__(self, vectorstore: VectorStore, top_k: int = 8):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    
    def retrieve_relevant_policies(self, query: str) -> List[Document]:
        """Retrieve and filter the most relevant policy documents."""
        # Get initial results
        initial_results = self.retriever.invoke(query)
        
        # Filter for relevance and diversity
        filtered_results = self._filter_and_diversify_results(initial_results, query)
        
        return filtered_results
    
    def _filter_and_diversify_results(self, documents: List[Document], query: str) -> List[Document]:
        """Filter results for relevance and ensure diversity of policy types."""
        if not documents:
            return documents
        
        # Group by policy type to ensure diversity
        policy_groups = {}
        for doc in documents:
            policy_type = doc.metadata.get('policy_type', 'Unknown')
            if policy_type not in policy_groups:
                policy_groups[policy_type] = []
            policy_groups[policy_type].append(doc)
        
        # Select best documents from each relevant policy type
        final_results = []
        query_lower = query.lower()
        
        # Define policy keywords for better matching
        policy_keywords = {
            'vacation': ['vacation', 'time off', 'pto', 'leave', 'holiday'],
            '401k': ['401k', 'retirement', 'pension', 'savings', 'matching'],
            'health': ['health', 'medical', 'insurance', 'coverage', 'benefits'],
            'childcare': ['childcare', 'daycare', 'child care', 'family', 'dependent'],
            'gym': ['gym', 'fitness', 'wellness', 'exercise', 'health club'],
            'tuition': ['tuition', 'education', 'learning', 'school', 'training'],
            'work from home': ['remote', 'work from home', 'telecommute', 'wfh', 'hybrid'],
            'life insurance': ['life insurance', 'death benefit', 'beneficiary']
        }
        
        # Score policy types by relevance to query
        policy_scores = {}
        for policy_type, docs in policy_groups.items():
            score = 0
            policy_lower = policy_type.lower()
            
            # Direct policy name match
            if any(keyword in policy_lower for keyword in query_lower.split()):
                score += 10
            
            # Keyword matching
            for keyword_group, keywords in policy_keywords.items():
                if keyword_group in policy_lower:
                    if any(keyword in query_lower for keyword in keywords):
                        score += 5
            
            # Content relevance (simple keyword matching)
            for doc in docs[:2]:  # Check first 2 docs from each policy
                content_lower = doc.page_content.lower()
                if any(keyword in content_lower for keyword in query_lower.split()):
                    score += 1
            
            policy_scores[policy_type] = score
        
        # Sort policy types by relevance score and select top documents
        sorted_policies = sorted(policy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take documents from most relevant policies
        for policy_type, score in sorted_policies:
            if score > 0:  # Only include policies with some relevance
                # Add up to 2 documents from each relevant policy type
                docs_to_add = policy_groups[policy_type][:2]
                final_results.extend(docs_to_add)
                
                # Limit total results
                if len(final_results) >= 5:
                    break
        
        return final_results[:5]  # Return top 5 most relevant documents
    
    def search_policies(self, query: str) -> Dict[str, Any]:
        """Search policies with improved filtering and return formatted results."""
        relevant_docs = self.retrieve_relevant_policies(query)
        formatted_context = self.format_retrieved_context(relevant_docs)
        
        return {
            "query": query,
            "relevant_documents": relevant_docs,
            "formatted_context": formatted_context,
            "num_sources": len(relevant_docs)
        }
    
    def format_retrieved_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        if not docs:
            return "No relevant policy information found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            policy_type = doc.metadata.get('policy', 'Unknown Policy')
            filename = doc.metadata.get('filename', 'unknown.pdf')
            content = doc.page_content.strip()
            
            # Clean up content
            content = content.replace('\\n', ' ').replace('  ', ' ')
            
            context_parts.append(f"[Source {i} - {policy_type} ({filename})]\\n{content}")
        
        return "\\n\\n".join(context_parts)

# Initialize the improved policy retriever
improved_retriever = ImprovedPolicyRetriever(vectorstore, top_k=8)

# ----------------------------------------------------------------------
#                    IMPROVED ACCURATE POLICY RESPONDER

class ImprovedAccuratePolicyResponder:
    """Enhanced policy responder with better accuracy and context handling."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)  # Zero temperature for maximum accuracy
        
    def generate_accurate_response(self, user_query: str, retrieved_documents: List[Document]) -> str:
        """Generate an accurate response with proper citations and context validation."""
        
        if not retrieved_documents:
            return "I couldn't find relevant policy information for your question. Please try rephrasing your question or contact HR directly for assistance."
        
        # Format the retrieved documents with better structure
        formatted_sources = []
        for i, doc in enumerate(retrieved_documents, 1):
            policy_name = doc.metadata.get('policy', 'Unknown Policy')
            filename = doc.metadata.get('filename', 'unknown.pdf')
            content = doc.page_content.strip()
            
            # Clean and format content
            content = self._clean_content(content)
            
            formatted_sources.append(f"===== SOURCE {i}: {policy_name} ({filename}) =====\\n{content}")
        
        sources_text = "\\n\\n".join(formatted_sources)
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR assistant providing accurate information about company benefits policies. 
            You must provide precise, helpful responses based ONLY on the provided policy documents.

            CRITICAL REQUIREMENTS:
            1. Only use information explicitly stated in the provided policy documents
            2. If the policy documents don't contain information to answer the question, say so clearly
            3. Always cite the specific policy document when providing information
            4. Provide specific details like eligibility requirements, timeframes, amounts, and procedures
            5. Use clear, simple language while maintaining complete accuracy
            6. If multiple policies apply, reference each one appropriately
            7. If there are conditions, exceptions, or limitations, state them clearly
            8. Do not make assumptions or inferences beyond what's explicitly stated

            RESPONSE FORMAT:
            - Start with a direct answer to the user's question
            - Provide specific details with citations
            - Use bullet points for multiple items or requirements
            - Include any important conditions or exceptions
            - Reference specific policy documents by name
            - Use streamlit markdown formatting to print it on streamlit front-end.
            - Use backslash symbol before $ value with no space between them.

            CITATION FORMAT: "According to the [Policy Name]..." or "As stated in the [Policy Name]..."

            If you cannot find relevant information in the provided documents, respond with:
            "I don't have information about [specific topic] in the available policy documents. Please contact HR for assistance with this question."
            """),
            
            ("user", """User Question: {user_query}

            Policy Documents Available:
            {sources_text}

            Please provide a comprehensive and accurate answer based on the policy information provided. 
            Include specific citations and details.""")
        ])
        
        response_chain = response_prompt | self.llm | StrOutputParser()
        
        try:
            response = response_chain.invoke({
                "user_query": user_query,
                "sources_text": sources_text
            })
            
            return response.strip()
            
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}. Please try again or contact HR for assistance."
    
    def _clean_content(self, content: str) -> str:
        """Clean and format document content for better processing."""
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Remove common OCR artifacts
        content = content.replace('\\n', ' ')
        content = content.replace('\\t', ' ')
        content = content.replace('  ', ' ')
        
        # Ensure reasonable length
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        return content.strip()
    
    def add_source_references(self, response: str, documents: List[Document]) -> str:
        """Add a clean source reference section to the response."""
        
        if not documents:
            return response
        
        # Create source references without duplicates
        source_refs = []
        seen_sources = set()
        
        for doc in documents:
            policy_name = doc.metadata.get('policy', 'Unknown Policy')
            filename = doc.metadata.get('filename', 'unknown.pdf')
            source_key = f"{policy_name} ({filename})"
            
            if source_key not in seen_sources:
                source_refs.append(source_key)
                seen_sources.add(source_key)
        
        # Add clean source section
        if source_refs:
            #response += "\n\n" + "=" * 50
            #response += "\n**Policy Documents Referenced:**"
            #for i, source in enumerate(source_refs, 1):
             #   response += f"\n {i}. {source}"
            None
        
        return response

# Initialize the improved responder
improved_responder = ImprovedAccuratePolicyResponder()

# ----------------------------------------------------------------------
#                      CONVERSATION MANAGER
class ConversationManager:
    """Manages conversation context and handles follow-up questions."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.conversation_history = []
        self.current_context = {}
    
    def add_to_history(self, user_query: str, response: str, context: Dict[str, Any] = None):
        """Add an interaction to conversation history."""
        interaction = {
            "timestamp": pd.Timestamp.now(),
            "user_query": user_query,
            "response": response,
            "context": context or {}
        }
        self.conversation_history.append(interaction)
        
        # Keep only last 10 interactions to manage context length
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def format_conversation_history(self) -> str:
        """Format conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation"
        
        history_text = "Previous conversation:\\n"
        for i, interaction in enumerate(self.conversation_history[-5:], 1):
            history_text += f"\nQ{i}: {interaction['user_query']}"
            history_text += f"\nA{i}: {interaction['response'][:200]}..."  # Truncate for brevity
        
        return history_text
    
    def handle_followup_question(self, 
                                user_query: str, 
                                policy_retriever: ImprovedPolicyRetriever,
                                simplifier: EnhancedJargonSimplifier) -> str:
        """Handle follow-up questions with conversation context."""
        
        followup_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR assistant handling follow-up questions about 
            company benefits. Use the conversation history to understand the context and 
            provide relevant, consistent answers.

            Guidelines:
            - Reference previous parts of the conversation when relevant
            - Maintain consistency with previous answers
            - If the question requires new policy information, indicate you'll search for it
            - If clarifying a previous answer, be more specific
            - Keep answers concise but complete"""),
            
            ("user", """Conversation History:
            {conversation_history}

            Current Question: {current_question}

            Relevant Policy Information:
            {policy_context}

            Please provide a helpful answer that considers the conversation context.""")
        ])
        
        # Get relevant policy information
        retrieval_result = policy_retriever.search_policies(user_query)
        
        # Simplify the retrieved policy text
        simplified_context = simplifier.simplify_with_context(
            retrieval_result['formatted_context'], 
            user_query
        )
        
        followup_chain = followup_prompt | self.llm | StrOutputParser()
        
        response = followup_chain.invoke({
            "conversation_history": self.format_conversation_history(),
            "current_question": user_query,
            "policy_context": simplified_context
        })
        
        # Add to history
        self.add_to_history(user_query, response, {
            "policy_sources": len(retrieval_result['relevant_documents']),
            "response_type": "followup"
        })
        
        return response
    
    def detect_followup_intent(self, user_query: str) -> bool:
        """Detect if the query is a follow-up question."""
        followup_indicators = [
            "what about", "and what", "also", "additionally", "furthermore",
            "can you explain", "what if", "how about", "in that case",
            "follow up", "more details", "elaborate", "clarify"
        ]
        
        query_lower = user_query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.current_context = {}

# Initialize conversation manager
conversation_manager = ConversationManager()

# ----------------------------------------------------------------------
#                      COMPLETE EXPLAINER

class ImprovedPolicyExplainerChatbot:
    """Enhanced Policy Explainer focused on accuracy and proper document retrieval."""
    
    def __init__(self, vectorstore: VectorStore):
        self.policy_retriever = ImprovedPolicyRetriever(vectorstore)
        self.accurate_responder = ImprovedAccuratePolicyResponder()
        self.conversation_manager = ConversationManager()
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query with improved accuracy and retrieval."""
        
        print(f"\\n Processing query: '{user_query}'")
        
        # Step 1: Retrieve relevant policy information with improved filtering
        print(" Retrieving relevant policies...")
        retrieval_result = self.policy_retriever.search_policies(user_query)
        
        if not retrieval_result['relevant_documents']:
            return {
                "query": user_query,
                "response": "I couldn't find relevant policy information for your question. Please try rephrasing your question or contact HR directly for assistance.",
                "sources_used": 0,
                "source_documents": []
            }
        
        # Step 2: Generate accurate response with improved processing
        print(" Generating accurate response...")
        accurate_response = self.accurate_responder.generate_accurate_response(
            user_query, 
            retrieval_result['relevant_documents']
        )
        
        # Step 3: Add source references
        final_response = self.accurate_responder.add_source_references(
            accurate_response,
            retrieval_result['relevant_documents']
        )
        
        # Step 4: Add to conversation history
        self.conversation_manager.add_to_history(
            user_query, 
            final_response,
            {"sources_used": retrieval_result['num_sources']}
        )
        
        return {
            "query": user_query,
            "response": final_response,
            "sources_used": retrieval_result['num_sources'],
            "source_documents": [doc.metadata.get('policy_type', 'Unknown') 
                               for doc in retrieval_result['relevant_documents']]
        }
    
    def handle_followup(self, user_query: str) -> str:
        """Handle follow-up questions with improved context."""
        print(f"\n Handling follow-up: '{user_query}'")
        
        # Get relevant policy information for follow-up
        retrieval_result = self.policy_retriever.search_policies(user_query)
        
        if not retrieval_result['relevant_documents']:
            return "I couldn't find relevant policy information for your follow-up question. Please try rephrasing or contact HR directly."
        
        # Generate response with improved processing
        response = self.accurate_responder.generate_accurate_response(
            user_query,
            retrieval_result['relevant_documents']
        )
        
        # Add source references
        final_response = self.accurate_responder.add_source_references(
            response,
            retrieval_result['relevant_documents']
        )
        
        # Update conversation history
        self.conversation_manager.add_to_history(user_query, final_response)
        
        return final_response
    
    def chat_session(self):
        """Start an interactive chat session with improved accuracy."""
        print(" Welcome to the TechLance Benefits Policy Explainer!")
        print("=" * 60)
        print("I provide accurate, well-cited information from your company's benefits policies.")
        print("All responses are based strictly on official policy documents.")
        print("\nType your questions about company benefits policies.")
        print("Type 'quit' to exit, 'clear' to start fresh.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n Thank you for using the Policy Explainer! Goodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.conversation_manager.clear_history()
                    print("Conversation history cleared!")
                    continue
                
                elif not user_input:
                    continue
                
                # Detect if it's a follow-up question
                if (len(self.conversation_manager.conversation_history) > 0 and 
                    self.conversation_manager.detect_followup_intent(user_input)):
                    response = self.handle_followup(user_input)
                    print(f"\n Assistant: {response}")
                else:
                    result = self.process_query(user_input)
                    print(f"\n Assistant: {result['response']}")
                    if result['sources_used'] > 0:
                        print(f"\n Sources consulted: {result['sources_used']}")
                
            except KeyboardInterrupt:
                print("\n \n Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                continue

# Initialize the improved chatbot system
chatbot = ImprovedPolicyExplainerChatbot(vectorstore)
