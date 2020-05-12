import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_and_sem(x , array, ax, color='b', array_axis=1,label='0',
    log_y=False, make_min_zero=[False,False]):
    
    mean = array.mean(axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis)
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis)
    
    if make_min_zero[0] is not False:
        bi, bv = get_in_range(x,make_min_zero[1])
        add = np.min(mean[bi])
    else:
        add = 0

    ax.fill_between(x, sem_plus-add, sem_minus-add, color=color, alpha=0.5)
    x = ax.plot(x,mean-add, '-',color=color,label=label)
    if log_y:
    	ax.set_yscale('log')
    return x, ax


def send_email(message_text, subject):
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("pkhanna104@gmail.com", "wrlyltbmcjwnpxil")

    message = "Subject: %s\n\n%s" % (subject, message_text)

    server.sendmail("pkhanna104@gmail.com", "pkhanna104@gmail.com", message)
    server.close()

def cuzicks_test(list_of_rts, ranking_to_test=[1, 2, 3, 4], twosided=True):
    #Do Cuzick's test for group (rank) trend: http://www.stata.com/manuals13/rnptrend.pdf
     
    print 'order: ', ranking_to_test
    args = []
    args1 = []
    for lfp_i, lfp in enumerate(ranking_to_test):
        x = list(list_of_rts[lfp_i])
        y = list(np.zeros((len(x)))+lfp_i)
 
        args.append(y)
        args1.append(x)
 
    Y = np.hstack((args)) 
    X = np.hstack((args1)) 
 
    #Rankings of X
    N = len(X)
    Rnk = scipy.stats.rankdata(X)
    R = np.zeros(( len(ranking_to_test), ))
    L = np.zeros(( len(ranking_to_test), ))
    L2 = np.zeros(( len(ranking_to_test), ))
    Ni = np.zeros(( len(ranking_to_test), ))
    T = np.zeros(( len(ranking_to_test), ))
 
    for li, lfp in enumerate(ranking_to_test):
         
        #Find labels equal to this label
        ix = np.nonzero(Y==lfp)[0]
 
        #Get rank fo data: 
        R[li] = np.sum(Rnk[ix])
 
        #Get wts
        L[li] = lfp*len(ix)
        L2[li] = lfp*lfp*len(ix)
        T[li] = lfp*R[li]
        Ni[li] = len(ix)
 
    print 'T', np.sum(T)
    e_t = 0.5*(N+1)*np.sum(L)
    var_t = (N*np.sum(L2) - (np.sum(L)**2))*((N+1)/12)
    se_t = np.sqrt( var_t )
    z = ( np.sum(T) - e_t ) / se_t
    print 'z', z
    if twosided:
        if z > 0:
            p = (1 - scipy.stats.norm.cdf(z))*2 #Two tailed test
        else:
            p = (scipy.stats.norm.cdf(z))*2 #Two-tailed test
    else:
        p = (1 - scipy.stats.norm.cdf(z))
    return p