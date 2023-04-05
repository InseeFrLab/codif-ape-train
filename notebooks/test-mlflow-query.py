import sys
sys.path.append('./src')

from fetch_model import fetch_model

# model = fetch_model("https://projet-ape-543061.user.lab.sspcloud.fr", "FastText-APE", "4")
model = fetch_model("https://projet-ape-543061.user.lab.sspcloud.fr", "test", "1")

query = {"query" : {'TEXT_FEATURE': ["LMNPPPP 64415 ape meubleee", "cac PORTH s,o"],
 'AUTO': [None, "C"],
 'NAT_SICORE': [None, "E29"],
 'SURF': [None, 2.0],
 'EVT_SICORE': [None,"E98"]},
 'k': 2}

print(model.predict(query))
