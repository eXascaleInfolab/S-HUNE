#include "mex.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"
#include "limits.h"

#define EXP_TABLE_SIZE 1000
// #define ALPHA_KATZ_TABLE_SIZE 20+1 // always add one because j start from 1 rather than 0
#define MAX_EXP 6
#define RAND_MULTIPLIER 25214903917
#define RAND_INCREMENT 11
// #define RAND_MULTIPLIER 1103515245 // change Linear congruential generator's parameters have negligible impact on runtime performance
// #define RAND_INCREMENT 12345

double *expTable;

long long *walk; // walk sequence n-n-n-n-n-n
double *emb_n; //node embedding


long long *neg_sam_table; // negative sampling table
long long dim_emb;
long long num_n;

long long num_w;
long long num_wl;
long long num_pos_sample;
double starting_alpha;
double alpha;
double num_neg;
long long table_size;


// double *counter;
long long num_threads;
double *alpha_Katz_Table;

// unsigned long next_random_max=0;

unsigned long getNextRand(unsigned long next_random){
    unsigned long next_random_return = next_random * (unsigned long) RAND_MULTIPLIER + RAND_INCREMENT;
//     if (next_random_return>next_random_max) {
//         next_random_max = next_random_return;
//         mexPrintf("next_random_max %lu\n",next_random_max);
//     }
    return next_random_return;
}

long long get_a_neg_sample(unsigned long next_random, long long target_e, long long word){
    long long target_n;
    unsigned long long ind;
//     while(1){
    ind = (next_random >> 16) % table_size;
    target_n = neg_sam_table[ind];
//         if ((target_n != target_e) && (target_n != word))
//         if ((target_n != target_e))
//             break;
//         next_random = getNextRand(next_random);
//     }
    return target_n;
}


double sigmoid(double f) {
    if (f >= MAX_EXP) return 1;
    else if (f <= -MAX_EXP) return 0;
    else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))];
}

int sample_pos_target_index(long long max_length, unsigned long next_random){
    double v_rand_uniform = (double)next_random/(double)(ULONG_MAX);
    int i;
    for (i=0; i<max_length; i++)
        if (v_rand_uniform < alpha_Katz_Table[i])
            break;
//     printf("sampled window size is %lld\n",i);
    return i+1;
}

int get_a_neg_sample_Kless1(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=num_neg){
        return 1;
    }else{
        return 0;
    }
}

void learn_a_pair(int flag, long long loc1, long long loc2)
{
    double f=0,g=0,a=0,tmp; //f2=0,
    
    for (int d=0;d<dim_emb;d++)
        f += emb_n[loc1+d] * emb_n[loc2+d];
    
//     g = 1/(1+exp(-f));
    g = sigmoid(f);
    a = (flag-g)*alpha;
    
//     printf("decay weight: %f\n",decay_weight);
    
    for (int d=0; d<dim_emb; d++){
        tmp = emb_n[loc2 + d];
        emb_n[loc2 + d] += a*emb_n[loc1 + d];
        emb_n[loc1 + d] += a*tmp;
//         emb_n[loc2 + d] += a*emb_n[loc1 + d];
//         emb_n[loc1 + d] += a*emb_temp[d];
    }
    
    
}


void normalize_embeddings(){
    long long loc_node;
    double norm;
    int i,d;
    for (i=0;i<num_n;i++) {
        loc_node = i*dim_emb;
        norm=0;
        for (d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d];
        for (d=0; d<dim_emb; d++) emb_n[loc_node+d] = emb_n[loc_node+d]/sqrt(norm);
    }
}
//
// void shownorm(){
//     long long loc_node;
//     double norm, norm_ave=0;
//     int i,d;
//     for (i=0;i<num_n;i++) {
//         loc_node = i*dim_emb;
//         norm=0;
//         for (d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d];
//         norm_ave = norm_ave+sqrt(norm)/num_n;
//     }
//     printf("average norm is: %f\n",norm_ave);
// }


void learn(void *id)
// void learn(int a_num)
{
    long long target_e,target_n,word;
//     double *emb_temp = (double *)mxMalloc(dim_emb*sizeof(double)); //a node embedding
//     double *emb_2 = (double *)mxMalloc(dim_emb*sizeof(double)); //a node embedding
//     int d;
    unsigned long next_random = (long) rand();
    
    long long ind_start = num_w/num_threads * (long long)id;
    long long ind_end = num_w/num_threads * ((long long)id+1);
//     long long ind_start = 0;
//     long long ind_end = num_w;
    long long ind_len = ind_end-ind_start;
    double progress=0,progress_old=0;
//     long long counter=0;
    long long loc_walk,loc_w,loc_e,loc_n;
//     long long ind_start = 0;
//     long long ind_end = num_w;
//     alpha = starting_alpha;
    mexPrintf("Thread %lld starts from walk %lld to %lld\n",(long long)id,ind_start,ind_end);
    
    for (int pp=0; pp<num_pos_sample; pp++){ //traverse 5 times the corpus as in deepwalk
        normalize_embeddings();
        for (int w=ind_start; w<ind_end; w++) {
            progress = ((pp*ind_len)+(w-ind_start)) / (double) (ind_len*num_pos_sample);
            if (progress-progress_old > 0.001) {
                alpha = starting_alpha * (1 - progress);
                if (alpha < starting_alpha * 0.001) alpha = starting_alpha * 0.001;
                progress_old = progress;
//                 normalize_embeddings(); // normalization, avoid super large embedding length, but should not be used for multi threads
//                 if( (long long) id == 0) {
//                     mexPrintf("current alpha is: %f; Progress %.0f%%\n", alpha, progress*100);
// //                     shownorm();
//                 }
            }
//
            
            loc_walk = w*num_wl;
            for (int i=0; i<num_wl; i++) {
                word = walk[loc_walk+i];
                loc_w = (word-1)*dim_emb;
                
//             printf("w,i,word: %lld,%d,%d \n",w,i,word);
                
                next_random = getNextRand(next_random);
                int j = sample_pos_target_index(num_wl, next_random);
                // consider left hand side
                if (i-j>=0) {
                    target_e = walk[loc_walk+i-j];
                    loc_e = (target_e-1)*dim_emb;
//                     double f_old=0,f_new=0;
//                     for (int d=0;d<dim_emb;d++)
//                         f_old = f_old + emb_n[word-1 + d*num_n] * emb_n[target_e-1 + d*num_n];
//                     printf("pair: %lld,%lld\n",word,target_e);
                    if (word!=target_e){
//                         for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_e+d];
//                         learn_a_pair(1, loc_w, loc_e, emb_temp); //LEARN WITH POSITIVE SAMPLES
                        learn_a_pair(1, loc_w, loc_e);
//                     for (int d=0;d<dim_emb;d++)
//                         f_new = f_new + emb_n[word-1 + d*num_n] * emb_n[target_e-1 + d*num_n];
//
//                     if (f_old > f_new)
//                         printf("ERROR for Gradient: pair: %d and %d; f_old and f_new: %f and %f \n",word,target_e,f_old, f_new);
                        
                        if (num_neg<1){
                            next_random = getNextRand(next_random);
                            if (get_a_neg_sample_Kless1(next_random)==1){
                                next_random = getNextRand(next_random);
                                target_n = get_a_neg_sample(next_random, target_e, word);
                                if ((target_n != target_e) && (target_n != word)) {
                                    loc_n = (target_n-1)*dim_emb;
//                                     for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_n+d];
                                    learn_a_pair(0, loc_w, loc_n);
//                                     learn_a_pair(0, loc_w, loc_n, emb_temp);
                                }
                            }
                        }else{
                            for (int n=0;n<num_neg;n++){
                                next_random = getNextRand(next_random);
                                target_n = get_a_neg_sample(next_random, target_e, word);
                                if ((target_n != target_e) && (target_n != word)) {
                                    loc_n = (target_n-1)*dim_emb;
//                                     for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_n+d];
//                                     learn_a_pair(0, loc_w, loc_n, emb_temp);
                                    learn_a_pair(0, loc_w, loc_n);
                                }
                            }
                        }
                    }
                }
                // consider right hand side ii+jj*2<=length(data)
                if (i+j<num_wl) {
                    target_e = walk[loc_walk+i+j];
                    loc_e = (target_e-1)*dim_emb;
//                     double f_old=0,f_new=0;
//                     for (int d=0;d<dim_emb;d++)
//                         f_old = f_old + emb_n[word-1 + d*num_n] * emb_n[target_e-1 + d*num_n];
//                     printf("pair: %lld,%lld\n",word,target_e);
                    if (word!=target_e) {
//                         for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_e+d];
                        learn_a_pair(1, loc_w, loc_e);
                        
//                     for (int d=0;d<dim_emb;d++)
//                         f_new = f_new + emb_n[word-1 + d*num_n] * emb_n[target_e-1 + d*num_n];
//
//                     if (f_old > f_new)
//                         printf("ERROR for Gradient: pair: %d and %d; f_old and f_new: %f and %f \n",word,target_e,f_old, f_new);
//
                        if (num_neg<1){
                            next_random = getNextRand(next_random);
                            if (get_a_neg_sample_Kless1(next_random)==1){
                                next_random = getNextRand(next_random);
                                target_n = get_a_neg_sample(next_random, target_e, word);
                                if ((target_n != target_e) && (target_n != word)) {
                                    loc_n = (target_n-1)*dim_emb;
//                                     for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_n+d];
                                    learn_a_pair(0, loc_w, loc_n);
                                }
                            }
                        }else{
                            for (int n=0;n<num_neg;n++){
                                next_random = getNextRand(next_random);
                                target_n = get_a_neg_sample(next_random, target_e, word);
                                if ((target_n != target_e) && (target_n != word)) {
                                    loc_n = (target_n-1)*dim_emb;
//                                     for (int d=0;d<dim_emb;d++) emb_temp[d] = emb_n[loc_n+d];
                                    learn_a_pair(0, loc_w, loc_n);
                                }
                            }
                        }
                    }
                }
                
            }
        }
    }
//     printf("counter (word=target_e) : %lld\n", counter);
    pthread_exit(NULL);
}



void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 8) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "8 inputs required.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                "1 output required.");
    }
    
//     int win_size; // context window size
//     double alpha=0.01; // SGD learning rate
//     double *emb_n; //entity embedding
//     double *emb_r; //relation embedding
//     long long *walk; // walk sequence e-r-e-...-r-e
//     long long *neg_sam_table; // negative sampling table
    
    walk = (long long *)mxGetData(prhs[0]); // read from file
    num_w = mxGetN(prhs[0]);
    num_wl = mxGetM(prhs[0]);
    
    emb_n = mxGetPr(prhs[1]);
    num_n = mxGetN(prhs[1]);
    dim_emb = mxGetM(prhs[1]);
    
    num_pos_sample = mxGetScalar(prhs[2]);
    starting_alpha = mxGetScalar(prhs[3]);
    num_neg = mxGetScalar(prhs[4]);
    
    neg_sam_table = (long long *)mxGetData(prhs[5]);
    table_size = mxGetM(prhs[5]);
    
    num_threads = mxGetScalar(prhs[6]);
    double alpha_Katz = mxGetScalar(prhs[7]);
//     plhs[0] = mxCreateDoubleMatrix(num_n,dim_emb,mxREAL);
//     plhs[1] = mxCreateDoubleMatrix(num_r,dim_emb,mxREAL);
//     double *out1 = mxGetPr(plhs[0]);
//     double *out2 = mxGetPr(plhs[1]);
//     for (int d=0; d<dim_emb; d++)
//         for (int i=0; i<num_n; i++)
//             out1[i+d*num_n] = emb_n[i+d*num_n];
//     for (int d=0; d<dim_emb; d++)
//         for (int i=0; i<num_r; i++)
//             out2[i+d*num_r] = emb_r[i+d*num_r];
    
    
//     plhs[0] = mxCreateSharedDataCopy(prhs[1]);
//     plhs[1] = mxCreateSharedDataCopy(prhs[2]);
//     plhs[0] = mxDuplicateArray(prhs[1]);
//     plhs[1] = mxDuplicateArray(prhs[2]);
//     plhs[1] = mxCreateDoubleScalar(0);
//     counter = mxGetPr(plhs[1]);
    /* call the computational routine */
    
    mexPrintf("walk number: %lld; walk length: %d\n",num_w, num_wl);
    mexPrintf("num of nodes: %lld; embedding dimension: %lld\n",num_n,dim_emb );
    mexPrintf("num_pos_sample: %lld\n",num_pos_sample);
    mexPrintf("learning rate: %f\n",starting_alpha);
    mexPrintf("negative sample number: %f\n",num_neg);
    mexPrintf("neg table size: %lld\n",table_size);
    mexPrintf("num_threads: %lld\n",num_threads);
    mexPrintf("alpha_Katz: %f\n",alpha_Katz);
//     printf("The maximum value of unsigned LONG = %lu\n", ULONG_MAX);
    fflush(stdout);
//     for (int i=0; i< num_wl; i++)
//         printf("walk %d: %lld\n",i, walk[i]);
    
    //initialize expTable
    expTable = (double *)mxMalloc((EXP_TABLE_SIZE + 1) * sizeof(double));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    //initialize alpha_Katz_Table as probability
    alpha_Katz_Table = (double *)mxMalloc((num_wl) * sizeof(double));
    for (int i=0; i<num_wl;i++) {
        if (i==0) alpha_Katz_Table[i]=1;
        else alpha_Katz_Table[i] = pow(alpha_Katz,i)+alpha_Katz_Table[i-1];
    }
    for (int i=0; i<num_wl;i++) alpha_Katz_Table[i] = alpha_Katz_Table[i]/alpha_Katz_Table[num_wl-1];
    
//     *counter =0; // not used anymore, before it was used to show the loss function.
    long long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, learn, (long long *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    
//      learn(0);
    
//     learn(walk, emb_n, emb_r, dim_emb, num_n, num_r, num_w, num_wl, win_size, alpha, num_neg, neg_sam_table, table_size, loss, margin);
    
    /* create the output matrix */
    plhs[0] = mxDuplicateArray(prhs[1]);
    
}






