from scipy.sparse import lil_matrix, csc_matrix, save_npz, load_npz

KG_raw=load_npz("/home/ubuntu/results/ontology/KG_raw.npz")
KG=KG_raw.tolil()

Jaccard=lil_matrix((133610,133610))

for i in range(0,133610):
    for j in range(i,133610):
        if i==j:
            continue
        else:
            u=KG.getrowview(i)
            v=KG.getrowview(j)
            set_u=set(u.nonzero()[1])
            set_v=set(u.nonzero()[1])
            comm=float(len(set_u&set_v))
            if not comm:
                continue
            jacc=float(len(set_u&set_v))/float(len(set_u|set_v))
            Jaccard[i,j]=jacc
            Jaccard[j,i]=jacc

save_npz("/home/ubuntu/results/ontology/Jaccard.npz",Jaccard.tocsc())