# Domain Note - Agriculture

## Title
AgriAssist: Multimodal Crop Disease Diagnosis and Localized Advisory System using Generative AI, RAG, and Agentic Reasoning

## Motivation
Agriculture is a critical sector in India, and many farmers depend on timely crop-health decisions to protect yield and income. Crop diseases can spread quickly, but access to expert diagnosis and reliable treatment guidance is limited in many rural areas.

## Real Problem
Farmers often struggle to identify crop diseases from visual symptoms and usually receive generic advice that may not match their crop, disease severity, or region. Existing tools often stop at image classification and do not provide grounded, localized, or safety-aware treatment support.

## Who It Affects
Small and marginal farmers, agricultural extension workers, agri-tech advisory platforms, and government agricultural support programs.

## Why Existing Solutions Fall Short
Most crop disease systems focus only on classifying an uploaded leaf image. They do not retrieve verified agricultural guidance, explain confidence, or adapt recommendations to Indian regional conditions.

## Research Gap
There is a gap for an integrated agricultural AI system that combines multimodal image diagnosis, RAG-based advisory generation, synthetic data generation for rare diseases, India-specific localization, and ethical safeguards.

## Proposed Approach
AgriAssist uses a vision model to detect crop disease from an image and provide a confidence score. A retrieval module searches agricultural knowledge sources. An advisory module generates farmer-friendly treatment and prevention steps. A localization layer adapts advice based on Indian region and crop conditions.

## Justification
This approach connects the technical model to the real agricultural problem. Multimodal AI supports disease diagnosis from images, RAG grounds the advice in agricultural knowledge, generative models address rare disease data scarcity, and the ethics layer improves trust and responsible use.
