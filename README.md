# Anas Tourari – Portfolio Data & ML Engineer  
**Ingénieur de données junior | Analyste | Stagiaire PFE 2026**  
[![Email](https://img.shields.io/badge/Email-meanastourari@gmail.com-blue?style=flat&logo=gmail)](mailto:meanastourari@gmail.com)
[![Phone](https://img.shields.io/badge/Phone-+212%20684094642-green?style=flat&logo=whatsapp)](tel:+212684094642)
[![Location](https://img.shields.io/badge/Location-Fès,%20Maroc-orange?style=flat&logo=map)](https://goo.gl/maps/...)
[![Portfolio](https://img.shields.io/badge/Portfolio-GitHub-black?style=flat&logo=github)](https://github.com/Anas-Tou)

---

## Profil

Étudiant en **Master 2 Exploration Informatique des Données et Décisionnel (EID2)** – double diplôme **Sorbonne Paris Nord (à distance) – USMBA Fès**.  
À la recherche d’un **stage PFE de 6 mois** pour finaliser mes études.

> **Motivé, adaptable, curieux** – doté d’un fort esprit d’équipe et de bonnes capacités d’analyse.  
> **Objectif** : contribuer à des projets innovants en **IA, data engineering et transformation numérique**.

---

## Formation Académique

| Diplôme | Établissement | Année |
|-------|---------------|-------|
| **Master 2 EID2** | Sorbonne Paris Nord – USMBA (double diplôme, à distance) | 2025–2026 |
| **Master 1 WISD** | Université Sidi Mohamed Ben Abdellah | 2024–2025 |
| **Licence Sciences Mathématiques et Informatiques** | Université Moulay Ismaïl | 2020–2024 |

---

## Expériences & Projets

### 1. **Chatbot RAG Interactif**  
**RAG (Retrieval-Augmented Generation) avec BGE + LLM**  
[GitHub](https://github.com/mohamed-bouchalkha/RAG-Chatbot-Project)

- Pipeline complet : **chunking → BGE embeddings → FAISS → LLM (Groq Llama3)**  
- Recherche sémantique en temps réel, **latence < 300ms**  
- Déploiement via **Docker**, interface web (React)  
- **Zéro hallucination** : réponses basées uniquement sur les documents sources

> *"Un chatbot académique intelligent qui répond en citant ses sources – comme un étudiant parfait."*

---

### 2. **Système de Recommandation d’Anime (Hybride)**  
**Content-Based + Collaborative Filtering simplifié**  
[GitHub](https://github.com/Anas-Tou/AnimeRecModelApiTrained)

- **TF-IDF** sur genres, **One-Hot** sur type, **StandardScaler** sur popularité  
- Matrice de similarité **sparse** (cosine) → **< 600 Mo RAM**  
- Prédiction de note via **moyenne pondérée des 20 meilleurs animes vus**  
- **F1-Score@10 = 0.68** | **Inférence < 50ms**  
- **Fuzzy matching** pour gérer les fautes (`Narutp` → `Naruto`)

> *"Recommande en 50ms sans GPU. Prêt pour 1M d’animes."*

---

### 3. **Agrégateur d’Offres de Stage avec Kafka**  
**Pipeline data en temps réel**  
[GitHub](https://github.com/Anas-Tou/Internship-Aggregator)

- **Scraping** (LinkedIn, Indeed) → **Kafka Producer**  
- **Déduplication** (hash MD5) → **PostgreSQL**  
- **FastAPI** + **React + Tailwind CSS** (cartes interactives)  
- **Docker Compose** : tout en 1 commande

> *"1000+ offres/jour, zéro doublon, zéro perte – même si un serveur plante."*

---

### 4. **Fine-tuning LLaMA avec LoRA**  
**Adaptation contextuelle sur données personnalisées**

- Réduction **×30 de la VRAM** vs full fine-tuning  
- Maintien de la précision, **+40% tokens/s** en inférence  
- Expérimentations sur **perplexity, vitesse, coût**

---

### 5. **Développeur R&D VR – Système Solaire**  
**Faculté des Sciences, Meknès – Février/Juin 2024**

- Modélisation 3D + interactions immersives (Unity)  
- Optimisation pour casques low-cost (Quest 2)  
- Navigation temps réel, narration dynamique

---

## Compétences Techniques

```mermaid
graph LR
    A[IA & Data Science] --> B[TensorFlow]
    A --> C[PyTorch]
    A --> D[Scikit-learn]
    A --> E[LangChain]
    A --> F[LLMs, RAG, NLP]
    
    G[Langages] --> H[Python]
    G --> I[Java, R, C, C#]
    
    J[Fullstack] --> K[React, Next.js]
    J --> L[FastAPI, Flask]
    J --> M[API REST]
    
    N[Data Engineering] --> O[SQL, PostgreSQL]
    N --> P[ETL, Kafka]
    N --> Q[NoSQL: MongoDB]
    
    R[BI & MLOps] --> S[Power BI, Tableau]
    R --> T[Docker, AWS, Git, CI/CD]
