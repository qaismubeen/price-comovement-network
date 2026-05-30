Price Co-Movement Network — Item-Level CPI Graph Analysis

This project builds a graph-based model of consumer price behavior across Pakistan using three years of real Consumer Price Index data from the Pakistan Bureau of Statistics. Every node in the network is a consumer item. Every edge means two items moved in price together in a consistent pattern across multiple cities at the same time.

The Dataset

Monthly retail price data for 51 consumer items across five Pakistani cities: Karachi, Lahore, Islamabad, Peshawar, and Quetta. Covers 2022, 2023, and 2024. Seven product categories: Food, Energy, Clothing, Housing, Education, Health, and Transport. After reshaping to long format the dataset has 9,180 records. Missing values are filled using forward-fill then backward-fill within each City-Item-Year group.

How the Network is Built

Raw prices vary too much between items to compare directly. A loaf of bread costs around 80 rupees while petrol costs over 200. Instead of raw prices, the model works with monthly price change vectors. For each item, city, and year, an 11-element vector is calculated from the difference between consecutive monthly prices.

Cosine similarity is then calculated between every pair of items within each city. A similarity at or above the threshold tau means two items moved in price in a similar pattern that month-over-month in that city. If enough cities agree (the city-count threshold K), an edge is added between those two items in the yearly graph.

The main parameter setting used throughout this project is tau = 0.7 and K = 2.

Graph Results

The three yearly graphs all include all 51 items as nodes.

2022: 409 edges, graph density 0.32
2023: 431 edges, graph density 0.34
2024: 471 edges, graph density 0.37

Edge count grew every year, meaning price movements across different items became more synchronized from 2022 to 2024. This matches the economic conditions in Pakistan where broad inflationary pressure pulled prices across unrelated product categories upward together.

Key Findings

Gas consistently ranked at the top of betweenness centrality across all three years with a score of 0.056 in both 2023 and 2024. This means Gas price changes acted as a bridge connecting energy items with food, housing, and clothing items across the entire network. When gas prices moved, the shock spread rapidly through the consumption basket.

284 edges remained stable across all three years. These represent structural relationships that held regardless of specific economic events: Diesel and Petrol tracking each other perfectly, staple food clusters (Rice, Sugar, Wheat Flour, Cooking Oil, Chicken), seasonal vegetable pairs (Onion, Tomato), and household cost clusters (House Rent, Medicine, Cotton Cloth).

Between-category edges outnumber within-category edges in all three years, at roughly 57 to 43 percent. This shows that price co-movement in Pakistan is driven more by broad economic shocks like currency devaluation and fuel price increases than by sector-specific factors.

Sensitivity analysis tested tau from 0.5 to 0.9 and K from 1 to 3. At tau = 0.5 and K = 1 the graph has 759 edges and is almost fully connected. At tau = 0.9 and K = 3 only 10 edges survive, representing only the most consistent price relationships in the entire dataset. The chosen parameters of tau = 0.7 and K = 2 produce a balanced graph that is neither too dense to read nor too sparse to be useful.

Centrality Highlights

2022 top item by degree centrality: Ladies Readymade Garments (0.640), driven by supply chain disruptions and textile input cost increases.

2023 top item by degree centrality: Fish (0.600), reflecting food price spikes from floods and import restrictions.

2024 top item by degree centrality: Furniture (0.660), with Potatoes, Vegetable Ghee, Eggs, and Fresh Fruits close behind.

Gas held the top betweenness centrality position in 2023 and 2024, confirming its role as the primary price shock bridge in the Pakistani consumption network.

Stack

Python, Pandas, NumPy, scikit-learn, NetworkX, Matplotlib

Data source: Pakistan Bureau of Statistics — pbs.gov.pk

To run:
python main.py
