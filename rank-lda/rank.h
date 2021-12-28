#ifndef RANK_TM_RANK_H
#define RANK_TM_RANK_H
//----------------------------------------------------------------------------------
#include <gsl/gsl_randist.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "utility.h"
#include "gtpoints.h"
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
struct RankOptions {
public:
    double sigma;
    double mu;
    double pi;
    int K;
    int I;
    int A;
    int P;
    int burn;
    string corpus;
    string srctopics;
    string key;
    bool use_key;
    DisplayOptions display;
    Model model;
    Perplexity perplexity;
    bool show_loglike;
    string gtpoints;
    bool use_gtpoints;
    string output_dir;
    unordered_set<int> save_points;
    double left;
    double right;
    double alpha;
    bool use_alpha;
    string psi;
    string vocab;
    RankOptions();
};
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
class Rank {
protected:
    Stats stats;
    GtPoints gt_points;
    vector<string> topic_labels;
    unordered_map<string, int> topic_id;
    vector<vector<int>> corpus;
    vector<vector<int>> corpus_test;
    int test_tokens;
    vector<int>* corpus_t;
    vector<int>* corpus_t_test;
    bool* hidden;
    vector<int> visible_topics;
    vector<vector<int>> ground_truth;
    double** psi;
    double* pr;
    double* pr_test;
    int** n_w;
    int** n_w_test;
    int** n_d;
    int** n_d_test;
    int* n_d_dot;
    int* n_w_dot;
    int* n_w_dot_test;
    int** n_t;
    int** n_t_test;
    double* norm;
    unordered_map<string, int> word_id;
    unordered_map<int, string> id_word;
    unordered_map<string, int> psi_id;
    unordered_map<int, string> id_psi;
    bool* ks_word;
    RankOptions options;
    void leftToRight(int doc, vector<double>& wordProbabilities);
    double evaluateLeftToRight();
    double importance();
    string Clean(double d);
    void Init_g_t();
    void Load_corpus();
    void Load_psi();
    void Load_deltas();
    void Init_random();
    void Update_n();
    int Sample_test(int doc, int token);
    int Sample(int doc, int token);
    void Calculate_theta();
    void Calculate_theta_test();
    void Calculate_phi();
    void Display_perplexity();
    void Display_loglike();
    void prune(int iter, int n_prune, int start_y);
    virtual void Hide_topic(int t);
    void Display_stats(int iter);
    void Populate_prob(int i, int t, int word, int doc, int start);
    void Populate_prob_test(int i, int t, int word, int doc, int start);
    virtual int Pop_sample(int word, int doc);
    virtual int Pop_sample_test(int word, int doc);
public:
    gsl_rng * RANDOM_NUMBER;
    double alpha;
    double beta;
    double sigma;
    double mu;
    double pi;
    int T, K, D, V, B, I, P, A;
    vector<vector<double>> theta_test;
    vector<vector<double>> theta;
    vector<vector<double>> phi;
    vector<vector<double>> deltas;
    double*** delta_pows;
    double** delta_pow_sums;
    double* lambdas;
    void Write_distributions(int iter = -1);
    Rank(RankOptions options);
    void load();
    virtual void gibbs();
    void save(int iter = -1);
    ~Rank();
};
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
#endif //RANK_TM_RANK_H
