from imports import *


def broadcast_unsqueeze(t, dim, size):
    return t.unsqueeze(dim).broadcast_to(t.shape[:dim] + (size,) + t.shape[dim:])

def powerset(iterable):
    l = list(iterable)
    return list(chain.from_iterable([combinations(l, r) for r in range(len(l)+1)]))

def mysum(a):
    v = a[0]
    for x in a[1:]:
        v += x
    return v

def allat(l, inds):
    return [l[i] for i in inds]

def allnotat(l, inds):
    return [l[i] for i in range(len(l)) if i not in inds]

def mean(a):
    return mysum(a) / len(a)

def unzip3(a):
    r = ([], [], [])
    for x in a:
        r[0].append(x[0])
        r[1].append(x[1])
        r[2].append(x[2])
    return r

def unzip(a):
    r = ([], [])
    for x in a:
        r[0].append(x[0])
        r[1].append(x[1])
    return r


def flatten(a):
    r = []
    for l in a:
        for x in l:
            r.append(x)
    return r


def iterate1(a):
    return [a[:, x] for x in range(a.shape[1])]


def iterate0(a):
    return [a[x] for x in range(a.shape[0])]


def atelsenone(l, inds):
    return [l[x] if x in inds else None for x in range(len(l))]


def eqelsenone(l, selects):
    return [l[x] if l[x] in selects else None for x in range(len(l))]


def describe(a, col_names):
    return list(f"{x[0]}:{x[1]:3.3},{x[2]:3.3},{x[3]:3.3}" for x in
                zip(col_names, iterate0(np.min(a, 0)), iterate0(np.average(a, 0)), iterate0(np.max(a, 0))) if
                x[0] is not None)

def carthesian_prod(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def save(array, filename):
    file = open(filename, "wb")
    array = np.ascontiguousarray(array).astype(dtype=np.dtype(np.single).newbyteorder("<"))
    file.write(mysum([x.to_bytes(4, byteorder="little", signed=False) for x in array.shape]))
    file.write(array.tobytes())

def log(string, p = "logs/main.txt"):
    os.makedirs(os.path.basename(p), exist_ok=True)
    open(p, "a").write(string)

def print_n_log(string, p = "logs/main.txt"):
    os.makedirs(os.path.basename(p), exist_ok=True)
    print(string)
    open(p, "a").write(string)
