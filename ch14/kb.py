"""A synthetic Northwind knowledge base: brand guidelines + product sheets, with questions and gold answers."""
import json, random
rng = random.Random(0)
BRAND = {
 "Voice and tone": ["Northwind writes in a warm, outdoorsy, concise voice.", "Never use exclamation marks in customer-facing copy.",
                    "Prefer concrete benefits over superlatives such as best or revolutionary.", "Address the reader as you and keep sentences under 20 words."],
 "Logo usage": ["The logo must have clear space equal to the height of the N on every side.", "Never place the logo on busy photographs without a solid backing panel.",
                "The minimum logo width is 24 pixels on screen and 12 millimeters in print.", "Do not rotate, stretch, or recolor the logo."],
 "Color palette": ["The primary color is Fjord Green, hex 2E5E4E.", "The secondary color is Birch White, hex F4F1EA.", "The accent color is Ember Orange, hex D9732B, used for calls to action only.",
                   "Text on Fjord Green must be Birch White for contrast."],
 "Typography": ["Headlines use the Tundra Sans typeface in semibold.", "Body copy uses Tundra Serif at 16 pixels or larger on screen.", "Never set body copy in all capitals."],
 "Photography": ["Photos show products in real outdoor use, not studio white backgrounds.", "People in photos should look natural and unposed.", "AI-generated images must be labeled and approved by a reviewer before publication."],
 "Social media": ["Posts include at most three hashtags, always including #Northwind.", "Reply to customer comments within one business day.", "Do not post product prices on social media; link to the store instead."],
 "Legal claims": ["Waterproof claims require an IPX rating or a hydrostatic head test result.", "Warranty language must say limited lifetime warranty, never lifetime guarantee.", "Sustainability claims must cite the certification body by name."],
}
FIRST = ["Trail", "Ridge", "Ember", "Summit", "Canyon", "Harbor", "Pine", "Juniper", "Cedar", "River"]
SECOND = ["Mug", "Tent", "Pack", "Gloves", "Lamp", "Stove", "Bottle", "Jacket"]
products = []
for f in FIRST:
    for s in rng.sample(SECOND, 3):
        name = f"{f}{s}"; w = rng.choice([180, 240, 320, 450, 900, 1300]); hrs = rng.choice([4, 6, 8, 12]); color = rng.choice(["Fjord Green", "Slate Grey", "Ember Orange", "Birch White"])
        mat = rng.choice(["recycled polyester", "stainless steel", "merino wool", "aluminium", "ripstop nylon"])
        products.append({"name": name, "facts": [f"The {name} weighs {w} grams.", f"The {name} is made from {mat}.", f"The {name} comes in {color}.",
                                                 f"The {name} carries a limited lifetime warranty.", f"Battery or heat retention for the {name} lasts {hrs} hours in typical use."],
                         "w": w, "mat": mat, "color": color, "hrs": hrs})
docs = [{"id": f"brand-{i}", "title": f"Brand guidelines: {sec}", "sections": [{"heading": sec, "sentences": sents}]} for i, (sec, sents) in enumerate(BRAND.items())]
docs += [{"id": f"product-{p['name']}", "title": f"Product sheet: {p['name']}", "sections": [{"heading": "Specifications", "sentences": p["facts"][:3]},
                                                                                        {"heading": "Warranty and use", "sentences": p["facts"][3:]}]} for p in products]
Q = []
bq = [("Can we use exclamation marks in ads?", "Never use exclamation marks", "brand-0", "Can our copy include an exclamation point?"),
      ("What is the hex code of the primary brand color?", "2E5E4E", "brand-2", "Which hex value is Fjord Green?"),
      ("How much clear space does the logo need?", "height of the N", "brand-1", "What padding should surround the logo mark?"),
      ("Which typeface is used for headlines?", "Tundra Sans", "brand-3", "What font do titles use?"),
      ("How many hashtags can a post have?", "at most three hashtags", "brand-5", "What is the hashtag limit for social posts?"),
      ("Can we post prices on social media?", "Do not post product prices", "brand-5", "Are prices allowed in our social posts?"),
      ("What do waterproof claims require?", "IPX rating", "brand-6", "What evidence backs a waterproof claim?"),
      ("How should warranty language be phrased?", "limited lifetime warranty", "brand-6", "What wording do we use for the guarantee?"),
      ("What is the minimum logo width on screen?", "24 pixels", "brand-1", "How small can the logo be on a website?"),
      ("Must AI-generated images be labeled?", "must be labeled", "brand-4", "Do synthetic photos need a disclosure?"),
      ("What color is used for calls to action?", "Ember Orange", "brand-2", "Which accent shade goes on buttons?"),
      ("What size is body copy on screen?", "16 pixels", "brand-3", "How large should paragraph text be online?")]
for q, a, d, para in bq:
    Q.append({"q": q, "answer": a, "doc": d, "kind": "brand"}); Q.append({"q": para, "answer": a, "doc": d, "kind": "brand-paraphrase"})
for p in rng.sample(products, 20):
    Q.append({"q": f"How much does the {p['name']} weigh?", "answer": f"{p['w']} grams", "doc": f"product-{p['name']}", "kind": "product"})
    Q.append({"q": f"What material is the {p['name']} made of?", "answer": p["mat"], "doc": f"product-{p['name']}", "kind": "product"})
for q in ["What is the return policy for international orders?", "Who is the CEO of Northwind?", "Does the RidgeKayak come in red?", "What is the price of the TrailMug?"]:
    Q.append({"q": q, "answer": None, "doc": None, "kind": "unanswerable"})
json.dump({"docs": docs, "questions": Q}, open("kb.json", "w"), indent=1)
print(len(docs), "documents,", sum(len(s["sentences"]) for d in docs for s in d["sections"]), "sentences,", len(Q), "questions:",
      {k: sum(q["kind"] == k for q in Q) for k in ["brand", "brand-paraphrase", "product", "unanswerable"]})
