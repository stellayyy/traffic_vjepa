import argparse, json
from ..data import load_array, to_tchw

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--layout",default=None); p.add_argument("--key",default=None); a=p.parse_args(); x=load_array(a.data,a.key); y=to_tchw(x,a.layout); print(json.dumps({"raw_shape":list(x.shape),"canonical_shape_TCHW":list(y.shape),"min":float(y.min()),"max":float(y.max()),"mean":float(y.mean()),"std":float(y.std())},indent=2))
if __name__ == "__main__": main()
