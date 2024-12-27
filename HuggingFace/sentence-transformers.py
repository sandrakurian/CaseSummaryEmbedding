# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

from transformers import AutoTokenizer, AutoModel
import torch

# Attempt to load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

print("Tokenizer loaded successfully!")
    
# Attempt to load model
print("Loading model...")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

print("Model loaded successfully!")


texts = [
    "French is a Romance language spoken by approximately 300 million people worldwide. It is an official language in 29 countries, including France, Canada, and several African nations. Known for its influence on art, philosophy, and diplomacy, French is often considered a language of culture and international relations.",
    "German is a West Germanic language spoken by over 90 million people, primarily in Germany, Austria, and Switzerland, as well as parts of Belgium and Luxembourg. It is the most widely spoken native language in Europe and is known for its precision and structure. German is the language of major thinkers like Karl Marx, Friedrich Nietzsche, and Albert Einstein, and its philosophical and scientific contributions have shaped Western thought. The language uses the Latin alphabet but features umlauts ( ,  ,  ) and the sharp S ( ), which affect pronunciation and meaning. German grammar can be challenging due to its case system (nominative, accusative, genitive, and dative), where nouns, articles, and adjectives change form depending on their role in a sentence. Additionally, German syntax often places verbs at the end of sentences, which can be tricky for learners. However, German is highly logical and consistent, with relatively few exceptions to its rules. It is also an essential language in the fields of engineering, philosophy, and classical music. German speakers enjoy a rich tradition in literature and arts, from the works of Goethe and Schiller to the compositions of Bach and Beethoven. As the primary language of business in much of Europe, German is highly valued in the global job market.",
    "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives. It values individual freedoms, participation, and equality. Common forms include direct democracy, where citizens vote on policies directly, and representative democracy, where they elect leaders to make decisions.",
    "A republic is a form of government where the country is considered a ""public matter,"" and the head of state is elected rather than being a hereditary monarch. In republics, the people elect representatives to make decisions on their behalf, and the government is accountable to the people and governed by laws. Unlike democracies, which directly empower citizens to participate in decision-making, republics often rely on a representative democracy where elected officials legislate on behalf of the citizens. A key feature of a republic is the rule of law, meaning that no one, not even government leaders, is above the law. This helps ensure fairness and justice in the system. Republics can vary in structure, ranging from parliamentary republics, like India, where the head of state is separate from the head of government, to presidential republics, like the United States, where the president serves as both the head of state and government. In a republic, the constitution plays an important role in establishing the powers of the government, protecting individual rights, and outlining the process for electing leaders. Republics are commonly seen as a form of government that emphasizes individual rights, fairness, and the active participation of citizens in the political process.",
    "Bats are the only mammals capable of sustained flight, making them one of the most fascinating and unique creatures in the animal kingdom. Belonging to the order Chiroptera, bats are incredibly diverse, with over 1,400 species found across nearly every part of the world, except in extreme desert and polar regions. They are often misunderstood and overlooked, but bats play a critical role in ecosystems through their behavior and feeding habits. Bats are generally classified into two major suborders: Megachiroptera, which includes the large fruit bats or ""flying foxes,"" and Microchiroptera, which consists of smaller insectivorous bats. Fruit bats, or flying foxes, are typically larger, with wingspans that can exceed 6 feet in some species, and they are primarily found in tropical regions. The majority of bat species, however, are insectivorous, feeding on a wide range of insects, including mosquitoes, moths, beetles, and flies. Some bat species also feed on nectar, fruit, small vertebrates, or even blood. One of the most remarkable aspects of bats is their ability to fly. Their wings are a modified form of the mammalian forelimb, consisting of a thin membrane of skin stretched between elongated finger bones. This unique wing structure allows bats to have highly maneuverable flight, enabling them to hover in place, fly at high speeds, and even navigate through complex environments, such as caves or dense forests. Their flight is powered by strong muscles and a flexible, lightweight skeleton, and their wing shape and size vary significantly depending on their species and the type of flight they perform.",
    "Dolphins are highly intelligent marine creatures known for their playful behavior and complex social structures. They belong to the Cetacea order, along with whales and porpoises, and are characterized by their streamlined bodies, which make them excellent swimmers. Dolphins use echolocation to navigate and hunt, emitting sound waves that bounce off objects, helping them locate prey and avoid obstacles. These creatures are often found in groups, or pods, and exhibit strong social bonds, working together to herd fish and protect one another from predators. They are known for their communication skills, using a variety of vocalizations and body language. Bottlenose dolphins, the most well-known species, are frequently seen in aquariums and are often trained to interact with humans, showcasing their intelligence and ability to learn complex tasks. Dolphins are carnivorous, feeding on fish, squid, and other marine organisms. In the wild, their diet varies depending on their habitat, with some species relying more heavily on fish, while others focus on squid. Dolphins are threatened by habitat destruction, pollution, and hunting in certain regions. Despite these challenges, many species of dolphins are protected by international conservation laws, and their populations are monitored to ensure their survival in the wild.",
    "Jellyfish are fascinating creatures that inhabit oceans worldwide, from shallow coastal waters to the deep sea. They are known for their gelatinous, translucent bodies and their ability to sting with specialized cells called nematocysts, which release venom to capture prey or defend against predators. Jellyfish are members of the phylum Cnidaria and are made up of about 95% water, giving them their characteristic soft, squishy appearance. Despite their simplicity, jellyfish have been around for more than 500 million years, making them one of the oldest living creatures. They lack a brain and central nervous system, instead relying on a simple nerve net to detect changes in their environment. Jellyfish feed on small fish, plankton, and other marine organisms, using their tentacles to capture and immobilize their prey. Some species, like the lion s mane jellyfish, can grow up to 8 feet in diameter, with tentacles that can stretch up to 100 feet long. Although jellyfish are important to the marine food web, their populations can fluctuate dramatically. In recent years, jellyfish blooms have increased in certain regions, often attributed to factors like overfishing, climate change, and pollution. While these blooms can disrupt ecosystems and local fisheries, jellyfish remain an integral part of the ocean s biodiversity.", 
    "Japanese cuisine is renowned for its fresh, seasonal ingredients and subtle flavors. Dishes like sushi, sashimi, and ramen highlight seafood, rice, and vegetables. The minimalist approach prioritizes ingredient quality, with light seasonings such as soy sauce, wasabi, and pickled ginger, emphasizing natural tastes and harmony in each dish.",
    "Indian cuisine is celebrated for its complex, flavorful dishes, shaped by a diverse cultural and regional history. Spices like cumin, coriander, turmeric, and garam masala give Indian food its signature aroma and taste. Northern India is known for hearty dishes like butter chicken, naan, and biryani, often rich with ghee and spices. In contrast, the south offers lighter meals featuring rice, coconut, and tamarind, such as dosas, idlis, and sambar. Vegetarianism is prevalent in Indian cuisine due to cultural and religious influences, with lentils, beans, and vegetables forming the base of many curries and stews. Meals typically consist of multiple components, including dal (lentil soup), vegetable curries, and pickles, paired with rice or flatbreads like roti or chapati. Yogurt also plays an important role, serving as a cooling side to spicy dishes or as a base for sauces like raita. Indian food celebrates diversity, with each region showcasing its own specialties that reflect local ingredients, traditions, and customs, offering a flavorful journey through the country's culinary heritage.", 
    "The Toyota Corolla is a reliable and fuel-efficient compact car that has been a global favorite for decades. Known for its low maintenance costs, smooth ride, and strong resale value, it offers various trim levels with modern tech features and advanced safety options, making it an excellent choice for everyday driving.", 
    "The Ford Mustang is an iconic American muscle car that has become a symbol of power, performance, and style since its debut in the 1960s. Known for its bold design and powerful engines, the Mustang has undergone several updates while maintaining its legacy. The 2024 model offers a range of engines, including a 2.3-liter turbocharged four-cylinder and a V8 engine, providing drivers with impressive horsepower and torque. The Mustang s aggressive styling, with its sleek lines and signature grille, is complemented by modern tech features such as a digital dashboard and an advanced infotainment system. Inside, the cabin offers a mix of comfort and performance, with options for leather seating and high-quality materials. The Mustang is also known for its performance handling, with rear-wheel drive and various driving modes that allow drivers to customize their driving experience. The Mustang s combination of classic muscle car heritage and contemporary performance makes it a favorite among car enthusiasts who value power, speed, and design. The 2024 Mustang offers more than just a car; it represents a lifestyle and a connection to a long history of American automotive excellence."
]

# Tokenize and encode the text
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # Assuming embeddings are taken from the last hidden state or pooled output
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling as an example

# Normalize embeddings for cosine similarity
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

# Compute similarities (dot product is equivalent to cosine similarity here)
similarity_matrix = torch.matmul(embeddings, embeddings.T)

# Loop through all pairs of texts and print similarities
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = similarity_matrix[i, j].item()
        print(f"Similarity between text {i + 1} and text {j + 1}: {similarity:.4f}")

'''
RESULTS
Similarity between text 1 and text 2: 0.5636
Similarity between text 1 and text 3: 0.6586
Similarity between text 1 and text 4: 0.4907
Similarity between text 1 and text 5: 0.0986
Similarity between text 1 and text 6: 0.4133
Similarity between text 1 and text 7: 0.1669
Similarity between text 1 and text 8: 0.5301
Similarity between text 1 and text 9: 0.3097
Similarity between text 1 and text 10: 0.5029
Similarity between text 1 and text 11: 0.3306
Similarity between text 2 and text 3: 0.4587
Similarity between text 2 and text 4: 0.4492
Similarity between text 2 and text 5: 0.1852
Similarity between text 2 and text 6: 0.3688
Similarity between text 2 and text 7: 0.1569
Similarity between text 2 and text 8: 0.4274
Similarity between text 2 and text 9: 0.2779
Similarity between text 2 and text 10: 0.3791
Similarity between text 2 and text 11: 0.2812
Similarity between text 3 and text 4: 0.6884
Similarity between text 3 and text 5: 0.0914
Similarity between text 3 and text 6: 0.4233
Similarity between text 3 and text 7: 0.1874
Similarity between text 3 and text 8: 0.5272
Similarity between text 3 and text 9: 0.2757
Similarity between text 3 and text 10: 0.4759
Similarity between text 3 and text 11: 0.3678
Similarity between text 4 and text 5: 0.1199
Similarity between text 4 and text 6: 0.4110
Similarity between text 4 and text 7: 0.1985
Similarity between text 4 and text 8: 0.4170
Similarity between text 4 and text 9: 0.3679
Similarity between text 4 and text 10: 0.4121
Similarity between text 4 and text 11: 0.3876
Similarity between text 5 and text 6: 0.3341
Similarity between text 5 and text 7: 0.2755
Similarity between text 5 and text 8: 0.0646
Similarity between text 5 and text 9: 0.1899
Similarity between text 5 and text 10: 0.1155
Similarity between text 5 and text 11: 0.1597
Similarity between text 6 and text 7: 0.4536
Similarity between text 6 and text 8: 0.3702
Similarity between text 6 and text 9: 0.2252
Similarity between text 6 and text 10: 0.3139
Similarity between text 6 and text 11: 0.3618
Similarity between text 7 and text 8: 0.2004
Similarity between text 7 and text 9: 0.1111
Similarity between text 7 and text 10: 0.1481
Similarity between text 7 and text 11: 0.1274
Similarity between text 8 and text 9: 0.6198
Similarity between text 8 and text 10: 0.5101
Similarity between text 8 and text 11: 0.3510
Similarity between text 9 and text 10: 0.3042
Similarity between text 9 and text 11: 0.2925
Similarity between text 10 and text 11: 0.4247
'''