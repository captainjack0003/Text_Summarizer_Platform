# Text_Summarizer_Platform
This code will Connect LLM models Back_end, and you can upload the documents and searching algoritms will extract only relevent information and send it to llm models


![image](https://github.com/captainjack0003/Text_Summarizer_Platform/assets/75877962/15aff214-8702-4e5c-8eb6-0ec5389098c3)


![image](https://github.com/captainjack0003/Text_Summarizer_Platform/assets/75877962/d2a6d6a2-00d9-400c-aab7-414de4e58b52)

![image](https://github.com/captainjack0003/Text_Summarizer_Platform/assets/75877962/6d647125-7fbb-4026-890b-921a5733ec6a)

SIMILARITY Search Techniques:
Cosine similarity and k-nearest neighbors (k-NN) are both concepts related to similarity search, but they operate in different ways.
1. *Cosine Similarity:*
   - *Definition:* Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It quantifies the similarity of two vectors by calculating the cosine of the angle between them. The range of cosine similarity is [-1, 1], where 1 indicates perfect similarity, 0 indicates no similarity, and -1 indicates perfect dissimilarity.
   - *Use in Similarity Search:* In the context of similarity search, cosine similarity is often used to compare the similarity between two vectors. It is commonly employed in natural language processing (NLP) for measuring the similarity between text documents, but it can be applied to any set of vectors in a high-dimensional space.
2. *K-Nearest Neighbors (k-NN):*
   - *Definition:* k-NN is a classification or regression algorithm that works by finding the k training samples closest to a given input point in the feature space. The "closeness" is typically determined by a distance metric, such as Euclidean distance or Manhattan distance.
   - *Use in Similarity Search:* In similarity search, k-NN is used to find the k most similar items or data points to a given query point. The similarity is often calculated using a distance metric, and the k-nearest neighbors are those with the smallest distances to the query point.
*Key Differences:*
   - *Metric:* Cosine similarity is a measure of the cosine of the angle between vectors, focusing on the direction rather than the magnitude. k-NN, on the other hand, relies on distance metrics (e.g., Euclidean distance) to determine proximity.
   - *Output:* Cosine similarity provides a similarity score between -1 and 1 for a pair of vectors. k-NN outputs the k nearest neighbors, typically represented as a list of data points or their indices.
   - *Application:* Cosine similarity is commonly used in information retrieval, text mining, and recommendation systems. k-NN is a versatile algorithm used in various fields, including classification, regression, and similarity search.
   - *Scalability:* Cosine similarity is computationally efficient for sparse vectors, making it suitable for high-dimensional data. k-NN can be computationally expensive, especially as the dataset size grows, since it involves calculating distances to all data points.

In summary, while cosine similarity measures the angle between vectors, k-nearest neighbors focuses on identifying the closest data points based on a distance metric. The choice between them depends on the specific requirements of the similarity search task and the characteristics of the data.
