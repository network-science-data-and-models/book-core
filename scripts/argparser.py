import random
import argparse

WHALES = ['orca', 'blue whale', 'fin whale', 'sei whale', 'beluga', 'narwhal', 'minke whale']
SHARKS = ['tiger shark', 'wobbegong', 'carpet shark', 'pyjama shark', 'great white shark']
parser = argparse.ArgumentParser()
parser.add_argument("--whales", help="how many whales do you want me to print?", type=int)
parser.add_argument("--sharks", help="how many sharks do you want me to print?", type=int)
parser.add_argument("--double", help="doubles the number of whales and sharks printed", type=bool)
args = parser.parse_args()

if args.whales:
    n_whales = args.whales
else:
    n_whales = 0

if args.sharks:
    n_sharks = args.sharks
else:
    n_sharks = 0
    
if args.double:
    n_whales *= 2
    n_sharks *= 2
    
return_whales = []
if n_whales > 0:
    for ix in range(n_whales):
        return_whales.append(random.choice(WHALES))

return_sharks = []
if n_sharks > 0:
    for ix in range(n_sharks):
        return_sharks.append(random.choice(SHARKS))
        
print(return_whales)
print(return_sharks)
