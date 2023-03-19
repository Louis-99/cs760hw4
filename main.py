import numpy as np
import io
import math

# number of data samples for training
N = 30
# number of languages
KL = 3
# number of language characters
KS = 27
# smooth argument
ALPHA = 0.5

def read_file(prefix, st, ed):
    ret = []
    for i in range(st, ed):
        with open(f'languageID/{prefix}{i}.txt', 'r') as f:
            s = filter(lambda x: x.isalpha() or x == ' ', f.read())
            ret.append(list(s))
    return ret

def flat(lsls):
    return [v for ls in lsls for v in ls]

def pred_label(vale, valj, vals):
    if vale > valj and vale > vals:
        return 'e'
    elif valj > vale and valj > vals:
        return 'j'
    else:
        return 's'

def prior_prob(doc_trg):
    return (len(doc_trg) + ALPHA) / (N + KL * ALPHA)

def cond_prob(doc_trg, a):    
    n1 = sum(1 for v in flat(doc_trg) if v == a)
    n2 = sum(len(s) for s in doc_trg)
    return (n1 + ALPHA) / (n2 + KS * ALPHA)
    
def main():
    # Q1
    print("Q1: ")
    doc_e = read_file('e', 0, 10)
    doc_j = read_file('j', 0, 10)
    doc_s = read_file('s', 0, 10)
    hatpe = prior_prob(doc_e)
    hatpj = prior_prob(doc_j)
    hatps = prior_prob(doc_s)
    print(f"{hatpe=}")
    print(f"{hatpj=}")
    print(f"{hatps=}")

    # Q2
    print("Q2: ")
    c = 'abcdefghijklmnopqrstuvwxyz '
    theta_e = [cond_prob(doc_e, ch) for ch in c]
    print(f"{theta_e=}")

    # Q3
    print("Q3:")
    theta_j = [cond_prob(doc_j, ch) for ch in c]
    theta_s = [cond_prob(doc_s, ch) for ch in c]
    print(f"{theta_j=}")
    print(f"{theta_s=}")

    # Q4
    print("Q4: ")
    doc_e10 = read_file('e', 10, 11)[0]
    wc_e10 = [sum(1 for ch in doc_e10 if ch == trg) for trg in c]
    print(f"{wc_e10=}")

    # Q5
    print("Q5: ")
    log_theta_e = [math.log(theta) for theta in theta_e]
    log_theta_j = [math.log(theta) for theta in theta_j]
    log_theta_s = [math.log(theta) for theta in theta_s]
    log_hatp_x_ye_e10 = sum(wc_e10[i]*log_theta_e[i] for i in range(KS))
    log_hatp_x_yj_e10 = sum(wc_e10[i]*log_theta_j[i] for i in range(KS))
    log_hatp_x_ys_e10 = sum(wc_e10[i]*log_theta_s[i] for i in range(KS))
    print(f"{log_hatp_x_ye_e10=}")
    print(f"{log_hatp_x_yj_e10=}")
    print(f"{log_hatp_x_ys_e10=}")

    # Q6
    print("Q5: ")
    log_hatpe = math.log(hatpe)
    log_hatpj = math.log(hatpj)
    log_hatps = math.log(hatps)
    log_hatp_ye_x_e10 = log_hatpe + log_hatp_x_ye_e10
    log_hatp_yj_x_e10 = log_hatpj + log_hatp_x_yj_e10
    log_hatp_ys_x_e10 = log_hatps + log_hatp_x_ys_e10
    print(f"{log_hatp_ye_x_e10=}")
    print(f"{log_hatp_yj_x_e10=}")
    print(f"{log_hatp_ys_x_e10=}")
    print(f"pred: {pred_label(log_hatp_ye_x_e10, log_hatp_yj_x_e10, log_hatp_ys_x_e10)}")


    # Q6
    print("Q6: ")

    def pred(s):
        wc = [sum(1 for ch in s if ch == trg) for trg in c]
        log_hatp_x_ye = sum(wc[i]*log_theta_e[i] for i in range(KS))
        log_hatp_x_yj = sum(wc[i]*log_theta_j[i] for i in range(KS))
        log_hatp_x_ys = sum(wc[i]*log_theta_s[i] for i in range(KS))
        log_hatp_ye_x = log_hatpe + log_hatp_x_ye
        log_hatp_yj_x = log_hatpj + log_hatp_x_yj
        log_hatp_ys_x = log_hatps + log_hatp_x_ys
        return pred_label(log_hatp_ye_x, log_hatp_yj_x, log_hatp_ys_x)

    doc_teste = read_file('e', 10, 20)
    doc_testj = read_file('j', 10, 20)
    doc_tests = read_file('s', 10, 20)
    pred_e = [pred(s) for s in doc_teste]
    pred_j = [pred(s) for s in doc_testj]
    pred_s = [pred(s) for s in doc_tests]
    print(f"ee: {sum(1 for v in pred_e if v == 'e')}")
    print(f"ej: {sum(1 for v in pred_e if v == 'j')}")
    print(f"js: {sum(1 for v in pred_e if v == 's')}")
    print(f"je: {sum(1 for v in pred_j if v == 'e')}")
    print(f"jj: {sum(1 for v in pred_j if v == 'j')}")
    print(f"es: {sum(1 for v in pred_j if v == 's')}")
    print(f"se: {sum(1 for v in pred_s if v == 'e')}")
    print(f"sj: {sum(1 for v in pred_s if v == 'j')}")
    print(f"ss: {sum(1 for v in pred_s if v == 's')}")




if __name__ == '__main__':
    main()