import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def f(x): return x * x

if __name__ == "__main__":
    with mp.Pool(2) as p:
        print(p.map(f, [1,2,3,4]))