# Customer Satisfaction model

**Predicting how a customer will feel about a product before they even ordered it**

---

## 🗒️ Table of contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Tech Stack](#tech-stack)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
5. [Routes and Data Endpoints](#routes-data-endpoints)
6. [Folder Structure](#folder-structure)
7. [Deployment](#deployment)
8. [Future Improvement](#future-improvement)
9. [License](#license)

---
## 📝 Project Overview
 
For a given customer's historical data, we are tasked to predict the review score for the next order or purchase. We will be using the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This dataset has information on 100,000 orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allow viewing charges from various dimensions: from order status, price, payment, freight performance to customer location, product attributes and finally, reviews written by customers. The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.