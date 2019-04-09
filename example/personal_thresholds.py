'''personal_thresholds.py - Joshua Wallace - Apr 2019

This stores the SNR threshold functions determined to work for my
M4 work.

'''

def pdm_cutoff(p):

    p1_y = 7.5
    p1_x = np.log10(.04)
    p2_y = 5.8
    p2_x = np.log10(.3)
    p3_y = 5.8
    p3_x = np.log10(1.5)
    p4_y = 5.1
    p4_x = np.log10(4.)
    p5_y = 4.5
    p5_x = np.log10(100.)

    m1 = (p2_y-p1_y)/(p2_x-p1_x)
    b1 = p1_y - m1*p1_x
    m2 = (p3_y-p2_y)/(p3_x-p2_x)
    b2 = p2_y - m2*p2_x
    m3 = (p4_y-p3_y)/(p4_x-p3_x)
    b3 = p3_y - m3*p3_x
    m4 = (p5_y-p4_y)/(p5_x-p4_x)
    b4 = p4_y - m4*p4_x
    
    logp = np.log10(p)
    if logp < p1_x:
        raise RuntimeError("Period too short")
    elif logp < p2_x:
        return m1*logp + b1
    elif logp < p3_x:
        return m2*logp + b2
    elif logp < p4_x:
        return m3*logp + b3
    elif logp < p5_x:
        return m4*logp + b4
    else:
        raise RuntimeError("Period too long")


def ls_cutoff(p):

    p1_y = 13.3
    p1_x = np.log10(.04)
    p2_y = 13.3
    p2_x = np.log10(.1)
    p3_y = 6.5
    p3_x = np.log10(3.5)
    p4_y = 4.5
    p4_x = np.log10(4)
    p5_y = 4.0
    p5_x = np.log10(100.)

    m1 = (p2_y-p1_y)/(p2_x-p1_x)
    b1 = p1_y - m1*p1_x
    m2 = (p3_y-p2_y)/(p3_x-p2_x)
    b2 = p2_y - m2*p2_x
    m3 = (p4_y-p3_y)/(p4_x-p3_x)
    b3 = p3_y - m3*p3_x
    m4 = (p5_y-p4_y)/(p5_x-p4_x)
    b4 = p4_y - m4*p4_x


    logp = np.log10(p)
    if logp < p1_x:
        raise RuntimeError("Period too short")
    elif logp < p2_x:
        return m1*logp + b1
    elif logp < p3_x:
        return m2*logp + b2
    elif logp < p4_x:
        return m3*logp + b3
    elif logp < p5_x:
        return m4*logp + b4
    else:
        raise RuntimeError("Period too long")


def bls_cutoff(p):

    p1_y = 19.2
    p1_x = np.log10(.04)
    p2_y = p1_y
    p2_x = np.log10(.09)
    p3_y = 12.5
    p3_x = np.log10(14.)
    p4_y = 10.
    p4_x = np.log10(15.)
    p5_y = 10.
    p5_x = np.log10(18.0)
    p6_y = 0.
    p6_x = np.log10(19.5)
    p7_y = 0.
    p7_x = np.log10(100.)

    m1 = (p2_y-p1_y)/(p2_x-p1_x)
    b1 = p1_y - m1*p1_x
    m2 = (p3_y-p2_y)/(p3_x-p2_x)
    b2 = p2_y - m2*p2_x
    m3 = (p4_y-p3_y)/(p4_x-p3_x)
    b3 = p3_y - m3*p3_x
    m4 = (p5_y-p4_y)/(p5_x-p4_x)
    b4 = p4_y - m4*p4_x
    m5 = (p6_y-p5_y)/(p6_x-p5_x)
    b5 = p5_y - m5*p5_x
    m6 = (p7_y-p6_y)/(p7_x-p6_x)
    b6 = p6_y - m6*p6_x

    logp = np.log10(p)
    if logp < p1_x:
        raise RuntimeError("Period too short")
    elif logp < p2_x:
        return m1*logp + b1
    elif logp < p3_x:
        return m2*logp + b2
    elif logp < p4_x:
        return m3*logp + b3
    elif logp < p5_x:
        return m4*logp + b4
    elif logp < p6_x:
        return m5*logp + b5
    elif logp < p7_x:
        return m6*logp + b6
    else:
        raise RuntimeError("Period too long")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    xes = np.linspace(0.05,90,10000)

    ls = [ls_cutoff(val) for val in xes]
    bls = [bls_cutoff(val) for val in xes]
    pdm = [pdm_cutoff(val) for val in xes]

    plt.plot(xes,ls,label='LS')
    plt.plot(xes,pdm,label='PDM')
    plt.plot(xes,bls,label='BLS')
    plt.xscale('log')

    plt.legend(loc='best')

    plt.savefig("temp.pdf")
