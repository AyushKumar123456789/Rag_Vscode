import json

data = [
    {"Question": "What are macronutrients?", "Answer": "Macronutrients are nutrients needed in large amounts, including carbohydrates, proteins, and fats. They provide energy and are essential for basic bodily functions.", "Relevance Score": 0.9},
    {"Question": "What is the main role of carbohydrates in the body?", "Answer": "Carbohydrates provide energy, support proper functioning of the nervous system, heart, and kidneys, and serve as building blocks for larger macromolecules.", "Relevance Score": 0.88},
    {"Question": "What is the difference between simple and complex carbohydrates?", "Answer": "Simple carbohydrates consist of one or two sugar units, while complex carbohydrates are long chains of simple sugars. Complex carbs provide sustained energy.", "Relevance Score": 0.85},
    {"Question": "How are lipids used by the body?", "Answer": "Lipids provide energy, store energy in fat cells, and serve as major components of cell membranes. They also protect organs and aid in temperature regulation.", "Relevance Score": 0.9},
    {"Question": "What is protein, and why is it important?", "Answer": "Proteins are composed of amino acids and provide structure to bones, muscles, and skin. They also play key roles in metabolic reactions.", "Relevance Score": 0.87},
    {"Question": "How does the body use water?", "Answer": "Water is crucial for transporting nutrients, removing waste, and maintaining body temperature. It constitutes over 60 percentage of body weight.", "Relevance Score": 0.9},
    {"Question": "What are micronutrients?", "Answer": "Micronutrients, including vitamins and minerals, are required in small amounts for bodily functions, such as enzyme functions and hormone production.", "Relevance Score": 0.88},
    {"Question": "What is the role of vitamin C?", "Answer": "Vitamin C acts as an antioxidant, aids in collagen formation, and supports the immune system. It is a water-soluble vitamin.", "Relevance Score": 0.85},
    {"Question": "What are the steps in the digestion process?", "Answer": "Digestion involves ingestion, mechanical and chemical breakdown, nutrient absorption, and waste elimination.", "Relevance Score": 0.9},
    {"Question": "What is the function of the liver in digestion?", "Answer": "The liver produces bile, which emulsifies fats for easier digestion. It also regulates nutrient levels in the bloodstream.", "Relevance Score": 0.86},
    {"Question": "How do prebiotics and probiotics benefit the gut?", "Answer": "Prebiotics stimulate beneficial bacteria growth, and probiotics add beneficial bacteria, aiding digestion and reducing symptoms like lactose intolerance.", "Relevance Score": 0.87},
    {"Question": "What is peristalsis?", "Answer": "Peristalsis is the wave-like muscle contractions that move food through the digestive tract. It plays a crucial role in the digestive process.", "Relevance Score": 0.88},
    {"Question": "How does the body absorb nutrients?", "Answer": "Nutrients are absorbed in the small intestine through villi and microvilli, which increase surface area and transport nutrients into the bloodstream.", "Relevance Score": 0.89},
    {"Question": "What is the role of bile in digestion?", "Answer": "Bile emulsifies fats, breaking them down into smaller droplets, which allows digestive enzymes to further break down lipids for absorption.", "Relevance Score": 0.88},
    {"Question": "How are amino acids used in the body?", "Answer": "Amino acids, the building blocks of proteins, are used for tissue repair, enzyme and hormone production, and other essential body functions.", "Relevance Score": 0.9},
    {"Question": "Why is fiber important in the diet?", "Answer": "Fiber aids in digestion, prevents constipation, and may reduce the risk of chronic diseases such as heart disease and diabetes.", "Relevance Score": 0.88},
    {"Question": "What is the function of sodium in the body?", "Answer": "Sodium helps maintain fluid balance, supports nerve transmission, and plays a role in muscle contraction.", "Relevance Score": 0.86},
    {"Question": "How is cholesterol used in the body?", "Answer": "Cholesterol is used to make vitamin D, steroid hormones, and bile acids, which are important for digestion and cellular function.", "Relevance Score": 0.85},
    {"Question": "What are antioxidants, and why are they important?", "Answer": "Antioxidants protect cells from damage caused by free radicals and reduce the risk of chronic diseases.", "Relevance Score": 0.87},
    {"Question": "How does the body regulate blood sugar levels?", "Answer": "Blood sugar is regulated by insulin and glucagon, which manage glucose storage and release from the liver.", "Relevance Score": 0.88},
    {"Question": "What are the essential fatty acids?", "Answer": "Essential fatty acids, like omega-3 and omega-6, cannot be synthesized by the body and must be obtained from the diet. They support heart and brain health.", "Relevance Score": 0.9},
    {"Question": "What is metabolic homeostasis?", "Answer": "Metabolic homeostasis is the balance of nutrient intake and energy expenditure to maintain bodily functions.", "Relevance Score": 0.89},
    {"Question": "What role do vitamins play in the immune system?", "Answer": "Vitamins, particularly A, C, and E, support immune health by enhancing white blood cell function and providing antioxidant protection.", "Relevance Score": 0.86},
    {"Question": "How does hydration affect bodily functions?", "Answer": "Proper hydration is essential for nutrient transport, waste elimination, and temperature regulation. Dehydration impairs these processes.", "Relevance Score": 0.9},
    {"Question": "What is the role of calcium in the body?", "Answer": "Calcium is essential for bone health, muscle contraction, nerve function, and blood clotting.", "Relevance Score": 0.88},
    {"Question": "How is iron used in the body?", "Answer": "Iron is needed for oxygen transport in blood, energy production, and immune function.", "Relevance Score": 0.87},
    {"Question": "What are fat-soluble vitamins?", "Answer": "Fat-soluble vitamins (A, D, E, K) are stored in body fat and liver and are essential for various bodily functions, such as vision and bone health.", "Relevance Score": 0.85},
    {"Question": "What is glycogen, and how is it used?", "Answer": "Glycogen is the stored form of glucose in muscles and the liver. It is used as a quick energy source during physical activity.", "Relevance Score": 0.88},
    {"Question": "How are proteins digested and absorbed?", "Answer": "Proteins are broken down into amino acids in the stomach and small intestine, where they are absorbed into the bloodstream.", "Relevance Score": 0.89},
    {"Question": "What is the importance of potassium?", "Answer": "Potassium helps regulate fluid balance, nerve signals, and muscle contractions, and counters the effects of sodium on blood pressure.", "Relevance Score": 0.86},
    {"Question": "What are common sources of dietary fiber?", "Answer": "Dietary fiber is found in fruits, vegetables, whole grains, and legumes. It supports digestive health and helps control blood sugar levels.", "Relevance Score": 0.87},
    {"Question": "How does the body use glucose?", "Answer": "Glucose is the primary source of energy for cells and is especially important for brain and muscle function.", "Relevance Score": 0.9},
    {"Question": "What is the function of the pancreas in digestion?", "Answer": "The pancreas produces enzymes and bicarbonate to neutralize stomach acid and aid in the digestion of carbohydrates, proteins, and fats.", "Relevance Score": 0.87},
    {"Question": "What is the role of vitamin D?", "Answer": "Vitamin D promotes calcium absorption, supports immune function, and helps maintain bone health.", "Relevance Score": 0.88},
    {"Question": "Why are whole foods considered better than supplements?", "Answer": "Whole foods provide a variety of nutrients, fiber, and beneficial compounds that supplements may lack.", "Relevance Score": 0.89},
    {"Question": "What are electrolytes, and why are they important?", "Answer": "Electrolytes, including sodium and potassium, maintain fluid balance, enable muscle contractions, and support nerve function.", "Relevance Score": 0.88},
    {"Question": "How are triglycerides used in the body?", "Answer": "Triglycerides store energy, insulate organs, and help absorb fat-soluble vitamins. They are the main type of fat in food and the body.", "Relevance Score": 0.9},
    {"Question": "What is nutrient density?", "Answer": "Nutrient density refers to the amount of essential nutrients a food provides relative to its calorie content. High nutrient density is desirable.", "Relevance Score": 0.87},
    {"Question": "What is the difference between HDL and LDL cholesterol?", "Answer": "HDL (high-density lipoprotein) is 'good' cholesterol that helps remove excess cholesterol, while LDL (low-density lipoprotein) is 'bad' cholesterol that can lead to plaque buildup in arteries.", "Relevance Score": 0.86},
    {"Question": "How does stress affect nutrition?", "Answer": "Stress can lead to poor eating habits, such as overeating or skipping meals, and can affect digestion and nutrient absorption.", "Relevance Score": 0.88}
]

# Save the data to a JSON file
with open('nutrition_data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Data saved to nutrition_data.json")
