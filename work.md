Hands-on Assignment: Build a RAG System Using LangChain

Problem statement

You are given a document describing the policies of a fictional company called TechNova Solutions (int the same folder)

Employees want to ask questions about these policies using a simple AI-powered question-answering system.

Your task is to build a Retrieval-Augmented Generation (RAG) system using LangChain that retrieves relevant information from the given document and uses an LLM to answer questions.
Tasks to complete

    Load the provided company-policy document using LangChain.

    Split the document into smaller chunks

    Generate embeddings for the chunks

    Store the embeddings in a vector store.

    Create a retriever that retrieves the three most relevant chunks for a given question.

    Use a LangChain prompt template and an LLM to generate an answer using the retrieved context.

    Test your system using the three questions given below.

Test questions

Question 1 - Direct retrieval

How many paid annual leave days does a full-time employee receive per year?

Question 2 - Combining information

An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?

Question 3 - Information not present in the document

What is the company's parental leave policy?

Your system should answer using the provided document. For Question 3, it should clearly state that the document does not provide this information instead of inventing a policy.
Submission requirements

Submit one Python file containing your complete implementation. Include the outputs produced for all three test questions as comments at the end of the file.

Do not submit your API key or any other secret credentials.

 


Submit one Python file containing your complete implementation. Include the outputs produced for all three test questions as comments at the end of the file.

Do not submit your API key or any other secret credentials.

