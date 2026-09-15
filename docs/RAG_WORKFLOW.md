# RAG Workflow

## Overview
OmniBrain uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from financial documents before generating an answer.

## Document Processing
Documents are processed and divided into meaningful chunks for embedding and retrieval.

## ChromaDB
The processed chunks are converted into vector representations and stored in ChromaDB for semantic similarity search.

## Search Agent
The Search Agent retrieves relevant document information from ChromaDB for qualitative queries.

## Agentic Workflow
The Supervisor analyzes the user's query and routes it to the appropriate specialized agent.

- Search Agent — document and vector retrieval
- SQL Agent — structured financial queries
- Vision Agent — charts and tables

## Response Generation
Retrieved information is passed to the synthesis stage to generate the final response.

## Contribution
My contribution focused on RAG and backend integration, including document processing, ChromaDB/vector retrieval, basic RAG, LLM integration, and API integration.
