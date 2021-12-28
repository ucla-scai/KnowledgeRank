#define _XOPEN_SOURCE 600
#include <iostream>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iomanip>
#include <limits>
#include <set>
#include <map>
#include "rank.h"
#include <algorithm>
#include <chrono>
#include <gsl/gsl_cdf.h>
#include <string>
#include <stdlib.h>
//----------------------------------------------------------------------------------
using namespace std;
using namespace std::chrono;
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
RankOptions::RankOptions() {
    left = 0.0;
    right = DBL_MAX;
    sigma = 0.3;
    mu = 0.7;
    pi = 1.0;
    K = 100;
    I = 1000;
    P = 3;
    burn = 100;
    corpus = "sampled_input_200_c.dat";
    srctopics = "src_topics_c.dat";
    model = src;
    A = 5;
    perplexity = none;
    use_gtpoints = false;
    output_dir = ".";
    use_key = false;
    show_loglike = false;
    use_alpha = false;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Load_corpus(){

    ifstream fin;
    fin.open(options.corpus, ios_base::in);
    vector<string> lines;

    for (string line; getline(fin, line);) {
        lines.push_back(line);
    }
    fin.close();

    vector<string> lines_test;
    if (options.perplexity != none) {
        int train = (int)(((double)lines.size()) * 0.8);
        int test = lines.size() - train;
        for (int i=0; i<test; i++) {
            lines_test.push_back(lines.back());
            lines.pop_back();
        }
        reverse(lines_test.begin(), lines_test.end());
    }

    unordered_set<string> vocab;
    int id = 0;
    for (int i=0; i<lines.size(); i++) {
        string line = lines[i];
        istringstream fstring(line);
        vector<int> document;
        string word;
        int token;
        while (fstring >> word) {
            unordered_set<string>::const_iterator iter = vocab.find(word);
            if (iter == vocab.end()) {
                vocab.insert(word);
                word_id[word] = id;
                id_word[id] = word;
                id++;
            }
            token = word_id[word];
            document.push_back(token);
        }
        corpus.push_back(document);
    }

    V = vocab.size();

    n_t = new int*[V];
    for (int i=0; i<V; i++) {
        n_t[i] = new int[corpus.size()];
        for (int j=0; j<corpus.size(); j++) {
            n_t[i][j] = 0;
        }
    }
    for (int i=0; i<corpus.size(); i++) {
        for (int j=0; j<corpus[i].size(); j++) {
            int token = corpus[i][j];
            n_t[token][i]++;
        }
    }

    if (options.perplexity != none) {
        test_tokens = 0;
        for (int i=0; i<lines_test.size(); i++) {
            string line = lines_test[i];
            istringstream fstring(line);
            vector<int> document;
            string word;
            int token;
            while (fstring >> word) {
                unordered_set<string>::const_iterator iter = vocab.find(word);
                if (iter != vocab.end()) {
                    token = word_id[word];
                    document.push_back(token);
                }
            }
            if (document.size() > 0) {
                corpus_test.push_back(document);
                test_tokens += document.size();
            }
        }
        n_t_test = new int*[V];
        for (int i=0; i<V; i++) {
            n_t_test[i] = new int[corpus_test.size()];
            for (int j=0; j<corpus_test.size(); j++) {
                n_t_test[i][j] = 0;
            }
        }
        for (int i=0; i<corpus_test.size(); i++) {
            for (int j=0; j<corpus_test[i].size(); j++) {
                int token = corpus_test[i][j];
                n_t_test[token][i]++;
            }
        }
    }

    D = corpus.size();
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Load_psi(){

    ifstream fin_vocab;
    vector<string> vocab;
    fin_vocab.open(options.vocab, ios_base::in);
    for (string line; getline(fin_vocab, line);) {
        istringstream fstring(line);
        string word;
        fstring >> word;
        int id = word_id[word];
        psi_id[word] = id;
        id_psi[id] = word;
        vocab.push_back(word);
    }
    fin_vocab.close();

    psi = new double*[B];
    for (int i=0; i<B; i++) {
        psi[i] = new double[V];
    }

    unordered_map<string, vector<double>> topic_distro;
    ifstream fin_psi;
    fin_psi.open(options.psi, ios_base::in);
    for (string line; getline(fin_psi, line);) {
        istringstream fstring(line);
        int i = 0;
        string topic;
        fstring >> topic;
        double prob;
        double distro[V];
        while (fstring >> prob) {
            string word = vocab[i];
            int id = psi_id[word];
            distro[id] = prob;
            i++;
        }
        for (int j=0; j<V; j++) {
            topic_distro[topic].push_back(distro[j]);
        }
    }
    fin_psi.close();

    for (int i=0; i<B; i++) {
        int b = i + K;
        string topic = topic_labels[b];
        for (int j=0; j<V; j++) {
            psi[i][j] = topic_distro[topic][j];
        }
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
double Rank::importance() {

    const int S = 100;

    double totalLogLikelihood = 0.0;
    double topic_alpha = ((double)visible_topics.size()) * alpha;
    double topic_alpha_g = gammaln(topic_alpha);
    double sum_alpha_g = 0.0;
    for (int j=0; j<visible_topics.size(); j++) {
        sum_alpha_g += gammaln(alpha);
    }
    double sum_gamma_alphas = topic_alpha_g - sum_alpha_g;

    for (int i=0; i<corpus_test.size(); i++) {
        double sums[corpus_test[i].size()];
        for (int j=0; j<corpus_test[i].size(); j++) {
            sums[j] = 0.0;
        }
        vector<vector<double>> q;
        vector<vector<double>> q_sum;
        for (int tok=0; tok<corpus_test[i].size(); tok++) {
            vector<double> topic_probs;
            for (int j=0; j<visible_topics.size(); j++) {
                int t = visible_topics[j];
                int w = corpus_test[i][tok];
                auto iter = id_word.find(w);
                if (iter == id_word.end()) {
                    continue;
                }
                double topic_prob = alpha * phi[j][w];
                sums[tok] += topic_prob;
                topic_probs.push_back(topic_prob);
            }
            q.push_back(topic_probs);
            q_sum.push_back(topic_probs);
        }
        for (int tok=0; tok<q.size(); tok++) {
            for (int j=0; j<q[tok].size(); j++) {
                q[tok][j] = q[tok][j] / sums[tok];
                double add = j == 0 ? 0.0 : q_sum[tok][j-1];
                q_sum[tok][j] = q[tok][j] + add;
            }
        }
        vector<vector<int>> samples;
        vector<vector<int>> samples_topic;
        for (int j=0; j<visible_topics.size(); j++) {
            vector<int> top_samples;
            for (int s=0; s<S; s++) {
                top_samples.push_back(0);
            }
            samples_topic.push_back(top_samples);
        }
        for (int tok=0; tok<corpus_test[i].size(); tok++) {
            vector<int> tok_samples;
            for (int s=0; s<S; s++) {
                double scale = q_sum[tok].back() * gsl_rng_uniform(RANDOM_NUMBER);
                int topic = 0;
                if (q_sum[tok][0] <= scale) {
                    int low = 0;
                    int high = q_sum[tok].size()-1;
                    while (low <= high) {
                        if (low == high - 1) { topic = high; break; }
                        int mid = (low + high) / 2;
                        if (q_sum[tok][mid] > scale) high = mid;
                        else low = mid;
                    }
                }
                tok_samples.push_back(topic);
                samples_topic[topic][s]++;
            }
            samples.push_back(tok_samples);
        }

        vector<double> log_pz;
        double token_topic_alpha_g = gammaln(((double)corpus_test[i].size()) + topic_alpha);
        for (int s=0; s<S; s++) {
            double sum = 0.0;
            for (int j=0; j<visible_topics.size(); j++) {
                sum += gammaln(samples_topic[j][s] + alpha);
            }
            log_pz.push_back(sum + sum_gamma_alphas - token_topic_alpha_g);
        }
        vector<double> log_w_given_z;
        for (int s=0; s<S; s++){
            log_w_given_z.push_back(0.0);
        }
        for (int tok=0; tok<corpus_test[i].size(); tok++) {
            int w = corpus_test[i][tok];
            for (int s = 0; s < S; s++) {
                int t = samples[tok][s];
                log_w_given_z[s] += log(phi[t][w]);
            }
        }
        vector<double> log_joint;
        vector<double> log_q;
        for (int s=0; s<S; s++) {
            log_joint.push_back(log_pz[s] + log_w_given_z[s]);
            log_q.push_back(0);
        }

        for (int tok=0; tok<corpus_test[i].size(); tok++) {
            for (int s = 0; s < S; s++) {
                int t = samples[tok][s];
                log_q[s] += log(q[tok][t]);
            }
        }
        vector<double> log_weights;
        for (int s=0; s<S; s++) {
            log_weights.push_back(log_joint[s] - log_q[s]);
        }
        totalLogLikelihood += (logsumexp(log_weights) - log(((double)S)));
    }

    return totalLogLikelihood;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
double Rank::evaluateLeftToRight() {
    double logNumParticles = log(10.0);
    double totalLogLikelihood = 0;
    for (int i=0; i<corpus_test.size(); i++) {

        double docLogLikelihood = 0;

        vector<vector<double>> particleProbabilities;
        for (int particle = 0; particle < 10; particle++) {
            vector<double> particleProbs;
            leftToRight(i, particleProbs);
            particleProbabilities.push_back(particleProbs);
        }

        for (int position = 0; position < particleProbabilities[0].size(); position++) {
            double sum = 0;
            for (int particle = 0; particle < 10; particle++) {
                sum += particleProbabilities[particle][position];
            }

            if (sum > 0.0) {
                double logProb = log(sum) - logNumParticles;
                docLogLikelihood += logProb;
            }
        }

        totalLogLikelihood += docLogLikelihood;
    }

    return totalLogLikelihood;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::leftToRight(int doc, vector<double>& wordProbabilities) {

    unordered_map<int, unordered_map<int, int>> typeTopicCounts;
    for (auto iter = id_word.begin(); iter != id_word.end(); ++iter) {
        int id = iter->first;
        unordered_map<int, int> topicCounts;
        for (int j=0; j<visible_topics.size(); j++) {
            int topic = visible_topics[j];
            int topicCount = n_w[topic][id];
            topicCounts[topic] = topicCount;
        }
        typeTopicCounts[id] = topicCounts;
    }

    int numTopics = visible_topics.size();

    double betaSums[numTopics];

    for (int j=0; j<numTopics; j++) {
        betaSums[j] = 0;
    }

    for (int j=0; j<numTopics; j++) {
        int t = visible_topics[j];
        if (t < K) {
            betaSums[j] = ((double) V) * beta;
        }
        else {
            int b = t - K;
            double sum = 0.0;
            for (int a = 0; a < A; a++) {
                double delta_a_j_sum = delta_pow_sums[b][a];
                sum += (((double) V) * delta_a_j_sum)*norm[a];
            }
            betaSums[j] = sum;
        }
    }

    double cachedCoefficients[numTopics];
    double smoothingOnlyMass[V];
    for (int j=0; j<V; j++) {
        smoothingOnlyMass[j] = 0.0;
    }

    for (int topic=0; topic < numTopics; topic++) {
        int t = visible_topics[topic];
        if (t < K) {
            for (int j=0; j<V; j++) {
                smoothingOnlyMass[j] += alpha * beta / (n_w_dot[t] + ((double) V) * beta);
            }
            cachedCoefficients[topic] =  alpha / (n_w_dot[t] + ((double) V) * beta);
        }
        else {
            int b = t-K;
            double sum = 0.0;
            double sum_smooth[V];
            for (int i=0; i<V; i++) {
                sum_smooth[i] = 0.0;
            }
            for (int a=0; a<A; a++) {
                double delta_a_j_sum = delta_pow_sums[b][a];
                for (int j=0; j<V; j++) {
                    double delta_i_j = delta_pows[b][j][a];
                    sum_smooth[j] += alpha * delta_i_j / (n_w_dot[t] + ((double) V) * delta_a_j_sum)*norm[a];
                }
                sum += alpha / (n_w_dot[t] + ((double) V) * delta_a_j_sum)*norm[a];
            }
            for (int j=0; j<V; j++) {
                smoothingOnlyMass[j] += sum_smooth[j];
            }
            cachedCoefficients[topic] =  sum;
        }
    }

    int oneDocTopics[corpus_test[doc].size()];

    wordProbabilities.clear();
    wordProbabilities.resize(corpus_test[doc].size());
    for (int j=0; j<corpus_test[doc].size(); j++) {
        wordProbabilities[j] = 0.0;
    }

    int type, newTopic;
    int docLength = corpus_test[doc].size();

    int tokensSoFar = 0;

    int localTopicCounts[numTopics];
    for (int j=0; j<numTopics; j++) {
        localTopicCounts[j] = 0;
    }
    int localTopicIndex[numTopics];

    int denseIndex = 0;

    int nonZeroTopics = denseIndex;

    double topicBetaMass = 0.0;
    double topicTermMass = 0.0;

    double topicTermScores[numTopics];
    int i;
    double score;

    double alphaSum = ((double) visible_topics.size()) * alpha;

    for (int limit = 0; limit < docLength; limit++) {
        type = corpus_test[doc][limit];

        auto iter = id_word.find(type);
        if (iter == id_word.end()) {
            continue;
        }

        int currentTopic, currentValue;

        topicTermMass = 0.0;

        for (int j = 0; j<visible_topics.size(); j++) {
            int index = visible_topics[j];
            currentTopic = j;
            currentValue = typeTopicCounts[type][index];

            score = cachedCoefficients[currentTopic] * currentValue;
            topicTermMass += score;
            topicTermScores[currentTopic] = score;
        }

        double sample = nextUniform(RANDOM_NUMBER) * (smoothingOnlyMass[type] + topicBetaMass + topicTermMass);
        double origSample = sample;

        wordProbabilities[limit] += (smoothingOnlyMass[type] + topicBetaMass + topicTermMass) / (alphaSum + tokensSoFar);
        tokensSoFar++;

        newTopic = -1;

        if (sample < topicTermMass) {
            i = -1;
            while (sample > 0) {
                i++;
                if (i >= numTopics) {
                    throw std::invalid_argument("error indexing greater than numTopics");
                }
                sample -= topicTermScores[i];
            }

            newTopic = i;
        }
        else {
            sample -= topicTermMass;
            if (sample < topicBetaMass) {
                sample /= beta;

                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                    int topic = localTopicIndex[denseIndex];
                    int t = visible_topics[topic];
                    sample -= localTopicCounts[topic] / (((double)n_w_dot[t]) + betaSums[topic]);

                    if (sample <= 0.0) {
                        newTopic = topic;
                        break;
                    }
                }
            }
            else {
                sample -= topicBetaMass;
                sample /= beta;

                newTopic = 0;
                int t = visible_topics[newTopic];
                sample -= alpha / (((double)n_w_dot[t]) + betaSums[newTopic]);

                while (sample > 0.0) {
                    newTopic++;
                    if (newTopic >= numTopics) {
                        throw std::invalid_argument("error indexing greater than numTopics");
                    }
                    t = visible_topics[newTopic];
                    sample -= alpha / (((double)n_w_dot[t]) + betaSums[newTopic]);
                }
            }

        }

        if (newTopic == -1) {
            cerr << "sampling error: " << origSample << " " << sample << " " << smoothingOnlyMass << " "
            << topicBetaMass << " " << topicTermMass;
            newTopic = numTopics-1;
        }

        oneDocTopics[limit] = newTopic;

        int map_topic = visible_topics[newTopic];
        if (map_topic < K) {
            topicBetaMass -= beta * localTopicCounts[newTopic] / (((double)n_w_dot[map_topic]) + betaSums[newTopic]);
        }
        else {
            int b = map_topic-K;
            double sum = 0.0;
            for (int a=0; a<A; a++) {
                double delta_i_j = delta_pows[b][type][a];
                sum += (delta_i_j * localTopicCounts[newTopic] / (((double)n_w_dot[map_topic]) + betaSums[newTopic]))*norm[a];
            }
            topicBetaMass -= sum;
        }

        localTopicCounts[newTopic]++;

        if (localTopicCounts[newTopic] == 1) {

            denseIndex = nonZeroTopics;

            while (denseIndex > 0 && localTopicIndex[denseIndex - 1] > newTopic) {

                localTopicIndex[denseIndex] =
                        localTopicIndex[denseIndex - 1];
                denseIndex--;
            }

            localTopicIndex[denseIndex] = newTopic;
            nonZeroTopics++;
        }

        cachedCoefficients[newTopic] = (alpha + localTopicCounts[newTopic]) / (((double)n_w_dot[map_topic]) + betaSums[newTopic]);

        if (map_topic < K) {
            topicBetaMass += beta * localTopicCounts[newTopic] / (((double)n_w_dot[map_topic]) + betaSums[newTopic]);
        }
        else {
            int b = map_topic-K;
            double sum = 0.0;
            for (int a=0; a<A; a++) {
                double delta_i_j = delta_pows[b][type][a];
                sum += (delta_i_j * localTopicCounts[newTopic] / (((double)n_w_dot[map_topic]) + betaSums[newTopic]))*norm[a];
            }
            topicBetaMass += sum;
        }
    }

    for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
        int topic = localTopicIndex[denseIndex];
        int t = visible_topics[topic];
        cachedCoefficients[topic] = alpha / (((double)n_w_dot[t]) + betaSums[topic]);
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Display_loglike() {

    Calculate_theta();
    Calculate_phi();

    double total_words = 0.0;
    double document_sum = 0.0;
    for (int d = 0; d < corpus.size(); d++) {
        total_words += (double) corpus[d].size();
        double word_sum = 0.0;
        for (int w = 0; w < V; w++) {
            double n_w_doc = (double) n_t[w][d];
            if (n_w_doc == 0.0) {
                continue;
            }
            double topic_sum = 0.0;
            for (int t = 0; t < visible_topics.size(); t++) {
                int k = visible_topics[t];
                topic_sum += phi[k][w] * theta[d][k];
            }
            double log_topic_sum = log(topic_sum);
            word_sum += log_topic_sum * n_w_doc;
        }
        document_sum += word_sum;
    }
    cout << currentDateTime() << "...Rank.gibbs - end iter...log-likelihood " << document_sum << endl;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Display_perplexity() {

    Calculate_theta_test();
    Calculate_phi();

    double perplex = 0.0;

    if (options.perplexity == lr) {
        double totalLogLike = evaluateLeftToRight();
        perplex = exp(-1 * totalLogLike / ((double) test_tokens));
    }
    if (options.perplexity == imp) {
        double totalLogLike = importance();
        perplex = exp(-1 * totalLogLike / ((double) test_tokens));
    }
    if (options.perplexity == heinrich) {
        double total_words = 0.0;
        double document_sum = 0.0;
        for (int d = 0; d < corpus_test.size(); d++) {
            total_words += (double) corpus_test[d].size();
            double word_sum = 0.0;
            for (int w = 0; w < V; w++) {
                double n_w_doc = (double) n_t_test[w][d];
                if (n_w_doc == 0.0) {
                    continue;
                }
                double topic_sum = 0.0;
                for (int t = 0; t < visible_topics.size(); t++) {
                    int k = visible_topics[t];
                    topic_sum += phi[k][w] * theta_test[d][k];
                }
                double log_topic_sum = log(topic_sum);
                word_sum += log_topic_sum * n_w_doc;
            }
            document_sum += word_sum;
        }
        perplex = exp(-1 * document_sum / total_words);
    }
    cout << currentDateTime() << "...Rank.gibbs - end iter...perplexity " << perplex << endl;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Display_stats(int iter) {
    double avg_src = stats.cnt_model > 0 ? stats.tot_model / ((double)stats.cnt_model) : 0;
    double avg_reg = stats.cnt_reg > 0 ? stats.tot_reg / ((double)stats.cnt_reg) : 0;
    cout << currentDateTime() << "...Rank.gibbs - end iter...iteration time " << stats.iteration_time << endl;
    cout << currentDateTime() << "...Rank.gibbs - end iter...avg iteration time " << ((double)stats.tot_iteration_time) / ((double)iter+1) << endl;
    cout << currentDateTime() << "...Rank.gibbs - end iter...avg src " << avg_src << endl;
    cout << currentDateTime() << "...Rank.gibbs - end iter...avg reg " << avg_reg << endl;
    if (options.use_key) {
        cout << currentDateTime() << "...Rank.gibbs - end iter...accuracy " << ((((double) stats.assign_correct) / ((double) stats.assign_total)) * 100.0) << endl;
        cout << currentDateTime() << "...Rank.gibbs - end iter...correct " << stats.assign_correct << endl;
        cout << currentDateTime() << "...Rank.gibbs - end iter...total " << stats.assign_total << endl;
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Hide_topic(int t) {
    visible_topics.erase(std::remove(visible_topics.begin(), visible_topics.end(), t), visible_topics.end());
    hidden[t] = true;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::load() {
    cout << currentDateTime() << "...Rank.load - Start\n";
    I = options.I;
    K = options.model == bijective ? 0 : options.K;
    sigma = options.sigma;
    A = sigma == 0.0 ? 1 : Max(options.A, 1);
    mu = options.mu;
    P = options.P;
    pi = options.pi;
    cout << currentDateTime() << "...Rank.load - Load corpus\n";
    Load_corpus();
    beta = ((double)200) / ((double)V);
    cout << currentDateTime() << "...Rank.load - Load deltas\n";
    Load_deltas();
    T = B + K;
    Load_psi();
    cout << currentDateTime() << "...Rank.load - Init counts\n";

    Init_random();
    Update_n();
    hidden = new bool[T];
    for (int i=0; i<T; i++) {
        hidden[i] = false;
        visible_topics.push_back(i);
    }
    alpha = options.use_alpha ? options.alpha : ((double)50) / ((double)visible_topics.size());
    pr = new double[T+B];
    if (options.perplexity != none) {
        pr_test = new double[T+B];
    }
    cout << currentDateTime() << "...Rank.load - Done\n";
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Init_g_t(){
    gt_points.load_saved(options.gtpoints);
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::save(int iter) {
    Calculate_theta();
    Calculate_phi();
    Write_distributions(iter);
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::prune(int iter, int n_prune, int start_y){
    int burn = this->options.burn;
    if (visible_topics.size() <= K) {
        return;
    }
    double m = ((double)(start_y))/((double)(burn-(3*I/4)));
    double b = start_y;
    int cur_y = visible_topics.size()-K;
    int des_y = m*((double)(iter-burn)) + b;
    if (cur_y <= des_y) {
        return;
    }
    int to_prune = cur_y - des_y;
    std::map<int, vector<int>> ordered;
    for (int i=0; i<T; i++) {
        if (hidden[i]) {
            continue;
        }
        int number_assign = 0;
        for (int j=0; j<D; j++) {
            number_assign += n_d[i][j];
        }
        ordered[number_assign].push_back(i);
    }
    bool keep[T];
    for (int i=0; i<T; i++) {
        keep[i] = false;
    }
    int idx = 0;
    for (auto o_itr = ordered.begin(); o_itr != ordered.end(); ++o_itr) {
        for (int i=0; i<o_itr->second.size(); i++) {
            keep[o_itr->second[i]] = idx >= to_prune;
            idx++;
        }
    }
    for (int i=0; i<T; i++) {
        if (hidden[i] || visible_topics.size() <= K) {
            continue;
        }
        if (!keep[i]) {
            cout << currentDateTime() << "...Rank.gibbs - pruning topic " << i << endl;
            Hide_topic(i);
        }
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::gibbs() {
    cout << currentDateTime() << "...Rank.gibbs - Start\n";
    int burn = this->options.burn;
    int start_y = T-K;
    int n_prune = ceil(((double)start_y) / ((double)(I-burn)));
    stats.tot_iteration_time = 0;
    for (int iter=0; iter <= I; iter++) {
        stats.assign_correct = 0;
        stats.assign_total = 0;
        auto start = high_resolution_clock::now();
        stats.tot_reg = 0;
        stats.cnt_reg = 0;
        stats.tot_model = 0;
        stats.cnt_model = 0;
        stats.iteration_time = 0;
        cout << currentDateTime() << "...Rank.gibbs - begin iter " << iter << "...topics " << visible_topics.size() << endl;
        if (iter > 0 && options.show_loglike) {
            Display_loglike();
        }
        if ((iter % 100 == 0 && iter > 0) || options.save_points.find(iter) != options.save_points.end() || iter == I) {
            if (options.perplexity != none) {
                Display_perplexity();
                Calculate_theta();
                if (options.save_points.find(iter) != options.save_points.end()) {
                    Write_distributions(iter);
                }
                else {
                    Write_distributions();
                }
            }
            else {
                if (options.save_points.find(iter) != options.save_points.end()) {
                    save(iter);
                }
                else {
                    save();
                }
            }
        }
        if (iter % burn == 0 && iter > 0 && options.model != bijective) {
            prune(iter, n_prune, start_y);
        }
        for (int doc=0; doc<D; doc++) {
            for (int token=0; token<corpus[doc].size(); token++) {
                corpus_t[doc][token] = Sample(doc, token);
            }
        }
        if (options.perplexity != none) {
            for (int doc = 0; doc < corpus_test.size(); doc++) {
                for (int token = 0; token < corpus_test[doc].size(); token++) {
                    corpus_t_test[doc][token] = Sample_test(doc, token);
                }
            }
        }
        stats.iteration_time = duration_cast<milliseconds>(high_resolution_clock::now()-start).count();
        stats.tot_iteration_time += stats.iteration_time;
        Display_stats(iter);
    }
    cout << currentDateTime() << "...Rank.gibbs - Done\n";
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class T> void convertFromString(T &value, const std::string &s) {
    std::stringstream ss(s);
    ss >> value;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Load_deltas(){

    cout << currentDateTime() << "...Rank.Load_deltas - begin\n";

    for (int i=0; i<K; i++) {
        topic_labels.push_back(to_string(i));
    }

    ifstream fin;
    fin.open(options.srctopics, ios_base::in);
    for (string line; getline(fin, line);) {
        istringstream fstring(line);
        string word;
        int count;

        string label;
        fstring >> label;
        topic_labels.push_back(label);

        vector<double> src_words(V);
        for (int i = 0; i < V; i++) {
            src_words[i] = epsilon;
        }

        while (fstring >> word >> count) {
            auto iter = word_id.find(word);
            if (!(iter == word_id.end())) {
                src_words[iter->second] += count;
            }
        }

        deltas.push_back(src_words);
    }

    for (int i=0; i<topic_labels.size(); i++) {
        string label = topic_labels[i];
        topic_id[label] = i;
    }

    B = deltas.size();

    if (options.use_gtpoints) {
        Init_g_t();
    }

    delta_pow_sums = new double*[B];

    for (int i=0; i<B; i++) {
        delta_pow_sums[i] = new double[A];
    }

    if (sigma == 0.0) {
        lambdas = new double[1];
    }
    else {
        lambdas = new double[A];
    }

    double left_max = options.use_gtpoints ? 0.0 : options.left;
    double right_min = options.use_gtpoints ? 1.0 : options.right;

    double left_bound = Max(mu - 2*sigma, left_max);
    double right_bound = Min(mu + 2*sigma, right_min);

    double interval = A < 2.0 ? 0.0 : (right_bound - left_bound) / ((double)(A-1));

    for (int a=0; a<A; a++) {
        lambdas[a] = left_bound + a*interval;
    }

    norm = new double[A];
    double sum = 0.0;
    for (int a=0; a<A; a++) {
        if (sigma == 0.0) {
            norm[a] = 1.0;
            sum += 1.0;
            break;
        }
        double x = lambdas[a];
        double density = Normal(x, mu, sigma);
        norm[a] = density;
        sum += density;
    }

    for (int a=0; a<A; a++) {
        norm[a] = norm[a]/sum;
    }

    delta_pows = new double**[B];

    for (int i=0; i<B; i++) {
        delta_pows[i] = new double*[V];
        double sum[A];
        for (int a = 0; a < A; a++) {
            sum[a] = 0.0;
        }
        for (int j=0; j<V; j++) {
            delta_pows[i][j] = new double[A];
            for (int a = 0; a < A; a++) {
                double mapped = options.use_gtpoints ? gt_points.map(i, lambdas[a]) : lambdas[a];
                double val = pow(deltas[i][j], mapped);
                delta_pows[i][j][a] = val;
                sum[a] += val;
            }
        }
        for (int a=0; a<A; a++) {
            delta_pow_sums[i][a] = sum[a];
        }
    }

    if (options.use_key) {
        ifstream fin_t;
        fin_t.open(options.key, ios_base::in);
        for (string line; getline(fin_t, line);) {
            istringstream fstring(line);
            string label;
            vector<int> doc_key;
            while (fstring >> label) {
                int id = topic_id[label];
                doc_key.push_back(id);
            }
            ground_truth.push_back(doc_key);
        }
        fin_t.close();
    }

    cout << currentDateTime() << "...Rank.Load_deltas - end\n";
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Init_random(){
    corpus_t = new vector<int>[D];
    for (int doc=0; doc<D; doc++) {
        vector<int> topics;
        for (int token=0; token<corpus[doc].size(); token++) {
            int t = Rand(RANDOM_NUMBER, 0, T-1);
            topics.push_back(t);
        }
        corpus_t[doc] = topics;
    }

    if (options.perplexity != none) {
        corpus_t_test = new vector<int>[corpus_test.size()];
        for (int doc = 0; doc < corpus_test.size(); doc++) {
            vector<int> topics;
            for (int token = 0; token < corpus_test[doc].size(); token++) {
                int t = Rand(RANDOM_NUMBER, 0, T - 1);
                topics.push_back(t);
            }
            corpus_t_test[doc] = topics;
        }
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Update_n(){
    n_w = new int*[T];
    n_w_dot = new int[T];
    n_d_dot = new int[T];
    n_d = new int*[T];
    if (options.perplexity != none) {
        n_w_test = new int*[T];
        n_d_test = new int*[T];
        n_w_dot_test = new int[T];
    }

    for (int i=0; i<T; i++) {
        n_w_dot[i] = 0;
        n_w[i] = new int[V];
        n_d_dot[i] = 0;
        if (options.perplexity != none) {
            n_w_dot_test[i] = 0;
            n_w_test[i] = new int[V];
            for (int j=0; j<V; j++) {
                n_w_test[i][j] = 0;
            }
        }
        for (int j=0; j<V; j++) {
            n_w[i][j] = 0;
        }
        n_d[i] = new int[D];
        if (options.perplexity != none) {
            n_d_test[i] = new int[corpus_test.size()];
        }
        for (int j=0; j<D; j++) {
            n_d[i][j] = 0;
        }
        if (options.perplexity != none) {
            for (int j=0; j<corpus_test.size(); j++) {
                n_d_test[i][j] = 0;
            }
        }
    }
    for (int doc=0; doc<D; doc++) {
        unordered_set<int> topics_in_doc;
        for (int token = 0; token < corpus[doc].size(); token++) {
            int t = corpus_t[doc][token];
            int w = corpus[doc][token];
            topics_in_doc.insert(t);
            n_w[t][w]++;
            n_d[t][doc]++;
            n_w_dot[t]++;
        }
        for (auto itr = topics_in_doc.begin(); itr != topics_in_doc.end(); ++itr) {
            n_d_dot[(*itr)]++;
        }
    }
    if (options.perplexity != none) {
        for (int doc = 0; doc < corpus_test.size(); doc++) {
            for (int token = 0; token < corpus_test[doc].size(); token++) {
                int t = corpus_t_test[doc][token];
                int w = corpus_test[doc][token];
                n_w_test[t][w]++;
                n_d_test[t][doc]++;
                n_w_dot_test[t]++;
            }
        }
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Populate_prob_test(int i, int t, int word, int doc, int start) {
    if (t < K) {
        pr_test[i] = ((((double)n_w[t][word]) + ((double)n_w_test[t][word]) + beta) / ( ((double)n_w_dot[t]) + ((double)n_w_dot_test[t]) + ((double) V) * beta)) *
                     (((double) n_d_test[t][doc] + alpha) / (((double) (corpus_test[doc].size() - 1)) + ((double) visible_topics.size()) * alpha));
    }
    else if (id_psi.find(word) == id_psi.end()) {
        int b = t-K;
        double sum = 0.0;
        for (int a=0; a<A; a++) {
            double delta_i_j = delta_pows[b][word][a];
            double delta_a_j_sum = delta_pow_sums[b][a];

            sum += (((((double)n_w[t][word]) + ((double)n_w_test[t][word]) + delta_i_j) / ( ((double) n_w_dot[t]) + ((double)n_w_dot_test[t]) + delta_a_j_sum)) *
                    (((double) n_d_test[t][doc] + alpha) / (((double) (corpus_test[doc].size() - 1)) + ((double) visible_topics.size()) * alpha)) *
                    norm[a]);
        }
        pr_test[i] = sum;
    }
    else {
        int b = t-K;
        pr_test[i] = psi[b][word] * (((double) n_d_test[t][doc] + alpha) / (((double) (corpus_test[doc].size() - 1)) + ((double) visible_topics.size()) * alpha));
    }

    if (i > start) {
        pr_test[i] += pr_test[i-1];
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Populate_prob(int i, int t, int word, int doc, int start) {
    if (t < K) {
        pr[i] = (((double) n_w[t][word] + beta) / (((double) n_w_dot[t]) + ((double) V) * beta)) *
                (((double) n_d[t][doc] + alpha) / (((double) (corpus[doc].size() - 1)) + ((double) visible_topics.size()) * alpha));

        stats.cnt_reg++;
        stats.tot_reg += pr[i];
    }
    else if (id_psi.find(word) == id_psi.end()) {
        int b = t-K;
        double sum = 0.0;
        for (int a=0; a<A; a++) {
            double delta_i_j = delta_pows[b][word][a];
            double delta_a_j_sum = delta_pow_sums[b][a];

            sum += ((((double) n_w[t][word] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum)) *
                    (((double) n_d[t][doc] + alpha) / (((double) (corpus[doc].size() - 1)) + ((double) visible_topics.size()) * alpha)) *
                    norm[a]);
        }

        pr[i] = sum;
        stats.cnt_model++;
        stats.tot_model += pr[i];
    }
    else {
        int b = t-K;
        pr[i] = psi[b][word] * (((double) n_d[t][doc] + alpha) / (((double) (corpus[doc].size() - 1)) + ((double) visible_topics.size()) * alpha));

        stats.cnt_model++;
        stats.tot_model += pr[i];
    }

    if (i > start) {
        pr[i] += pr[i-1];
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Rank::Pop_sample_test(int word, int doc) {
    for (int i=0; i<visible_topics.size(); i++) {
        int t = visible_topics[i];
        Populate_prob_test(i, t, word, doc, 0);
    }
    double scale = pr_test[visible_topics.size()-1] * gsl_rng_uniform(RANDOM_NUMBER);
    int topic = 0;
    if (pr_test[0] <= scale) {
        int low = 0;
        int high = visible_topics.size()-1;
        while (low <= high) {
            if (low == high - 1) { topic = high; break; }
            int mid = (low + high) / 2;
            if (pr_test[mid] > scale) high = mid;
            else low = mid;
        }
    }
    return topic;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Rank::Pop_sample(int word, int doc) {
    for (int i=0; i<visible_topics.size(); i++) {
        int t = visible_topics[i];
        Populate_prob(i, t, word, doc, 0);
    }
    double scale = pr[visible_topics.size()-1] * gsl_rng_uniform(RANDOM_NUMBER);
    int topic = 0;
    if (pr[0] <= scale) {
        int low = 0;
        int high = visible_topics.size()-1;
        while (low <= high) {
            if (low == high - 1) { topic = high; break; }
            int mid = (low + high) / 2;
            if (pr[mid] > scale) high = mid;
            else low = mid;
        }
    }
    return topic;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Rank::Sample_test(int doc, int token){
    int topic = corpus_t_test[doc][token];
    int w = corpus_test[doc][token];
    n_d_test[topic][doc] = Max(n_d_test[topic][doc]-1, 0);
    n_w_test[topic][w] = Max(n_w_test[topic][w]-1, 0);
    n_w_dot_test[topic] = Max(n_w_dot_test[topic]-1, 0);
    alpha = options.use_alpha ? options.alpha : ((double)50) / ((double)visible_topics.size());
    topic = Pop_sample_test(w, doc);
    topic = visible_topics[topic];
    n_d_test[topic][doc]++;
    n_w_test[topic][w]++;
    n_w_dot_test[topic]++;
    return topic;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Rank::Sample(int doc, int token){
    int topic = corpus_t[doc][token];
    int w = corpus[doc][token];
    n_d[topic][doc] = Max(n_d[topic][doc]-1, 0);
    n_w[topic][w] = Max(n_w[topic][w]-1, 0);
    n_w_dot[topic] = Max(n_w_dot[topic]-1, 0);
    if (n_d[topic][doc] == 0) {
        n_d_dot[topic]--;
    }
    if (n_w_dot[topic] == 0 && visible_topics.size() > K && !hidden[topic] && options.model != bijective) {
        cout << currentDateTime() << "...removing topic " << topic << endl;
        Hide_topic(topic);
    }
    alpha = options.use_alpha ? options.alpha : ((double)50) / ((double)visible_topics.size());
    topic = Pop_sample(w, doc);
    topic = visible_topics[topic];
    stats.assign_total++;
    int offset = options.model == bijective ? 0 : K;
    if (options.use_key && ground_truth[doc][token] + offset == topic) {
        stats.assign_correct++;
    }
    if (n_d[topic][doc] == 0) {
        n_d_dot[topic]++;
    }
    n_d[topic][doc]++;
    n_w[topic][w]++;
    n_w_dot[topic]++;
    return topic;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Calculate_theta_test(){
    for (int i=0; i<theta_test.size(); i++) {
        theta_test[i].clear();
    }
    theta_test.clear();
    int topic_count = visible_topics.size();
    for (int doc=0; doc<corpus_test.size(); doc++) {
        vector<double> theta_d(T);
        for (int t=0; t<T; t++) {
            theta_d[t] = (((double)n_d_test[t][doc]) + alpha) / (((double)corpus_test[doc].size()) + (((double)topic_count)*alpha));
        }
        theta_test.push_back(theta_d);
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Calculate_theta(){
    for (int i=0; i<theta.size(); i++) {
        theta[i].clear();
    }
    theta.clear();
    int topic_count = visible_topics.size();
    for (int doc=0; doc<D; doc++) {
        vector<double> theta_d(T);
        for (int t=0; t<T; t++) {
            theta_d[t] = (((double)n_d[t][doc]) + alpha) / (((double)corpus[doc].size()) + (((double)topic_count)*alpha));
        }
        theta.push_back(theta_d);
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Calculate_phi(){
    for (int i=0; i<phi.size(); i++) {
        phi[i].clear();
    }
    phi.clear();
    for (int t=0; t<T; t++) {
        vector<double> phi_t(V);
        if (t < K) {
            for (int w=0; w<V; w++) {
                phi_t[w] = (((double)n_w[t][w]) + beta)/(((double)n_w_dot[t]) + (((double)V)*beta));
            }
        }
        else {
            int b = t-K;
            for (int w=0; w<V; w++) {
                if (id_psi.find(w) == id_psi.end()) {
                    double sum = 0.0;
                    for (int a = 0; a < A; a++) {
                        double delta_i_j = delta_pows[b][w][a];
                        double delta_a_j_sum = delta_pow_sums[b][a];
                        sum += (((double) n_w[t][w] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum)) * norm[a];
                    }
                    phi_t[w] = sum;
                }
                else {

                }
            }
        }
        phi.push_back(phi_t);
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
string Rank::Clean(double d) {

    std::stringstream ss(stringstream::in | stringstream::out);
    if (options.display.truncated) {
        ss << setprecision(options.display.d) << d;
    }
    else {
        ss << d;
    }
    string s = ss.str();
    return s;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Rank::Write_distributions(int iter) {

    int top = options.display.top ? options.display.n : T;
    vector<int> idx(T);

    string to_add = iter < 0 ? "" : "_" + to_string(iter);

    ofstream theta_out;
    theta_out.open(options.output_dir + "/theta" + to_add + ".dat");
    for (int doc=0; doc<D; doc++) {
        theta_out << doc;
        if (options.display.top) {
            vector<double> theta_d(theta[doc]);
            Sort(theta_d, idx);
            for (int t=0; t<Min(options.display.n, T); t++) {
                string topic = options.display.labels ? topic_labels[idx[t]] : to_string(idx[t]);
                theta_out << " " << topic << ":" << Clean(theta_d[t]);
            }
        }
        else {
            for (int t=0; t<T; t++) {
                theta_out << " " << Clean(theta[doc][t]);
            }
        }
        theta_out << endl;
    }
    theta_out.close();

    ofstream phi_out;
    phi_out.open(options.output_dir + "/phi" + to_add + ".dat");
    for (int t=0; t<T; t++) {
        if (hidden[t]) { continue; }
        string topic = options.display.labels ? topic_labels[t] : to_string(t);
        phi_out << topic;
        if (options.display.top) {
            vector<double> phi_t(phi[t]);
            Sort(phi_t, idx);
            for (int w=0; w<Min(options.display.n, V); w++) {
                string word = options.display.labels ? id_word[idx[w]] : to_string(idx[w]);
                phi_out << " " << word << ":" << Clean(phi_t[w]);
            }
        }
        else {
            for (int w=0; w<V; w++) {
                phi_out << " " << Clean(phi[t][w]);
            }
        }
        phi_out << endl;
    }
    phi_out.close();

    ofstream v_out;
    v_out.open(options.output_dir + "/vocab_tm" + to_add + ".dat");
    for (auto itr = word_id.begin(); itr != word_id.end(); ++itr) {
        v_out << itr->first << " " << itr->second << endl;
    }
    v_out.close();

    ofstream t_out;
    t_out.open(options.output_dir + "/corpus_t" + to_add + ".dat");
    for (int doc=0; doc<D; doc++) {
        for (int t=0; t<corpus_t[doc].size(); t++) {
            int cur = corpus_t[doc][t];
            string topic = options.display.labels ? topic_labels[cur] : to_string(cur);
            if (t==0) {
                t_out << topic;
            }
            else {
                t_out << " " << topic;
            }
        }
        t_out << endl;
    }
    t_out.close();
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Rank::Rank(RankOptions options) {
    this->options = options;
    RANDOM_NUMBER = gsl_rng_alloc(gsl_rng_taus);
    time_t t;
    time(&t);
    long seed = (long) t;
    gsl_rng_set(RANDOM_NUMBER, (long) seed);
    if (options.output_dir.back() == '/') {
        this->options.output_dir = options.output_dir.substr(0, options.output_dir.size()-1);
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Rank::~Rank() {
    for (int i=0; i<T; i++) {
        delete[] n_w[i];
        delete[] n_d[i];

        if (options.perplexity != none) {
            delete[] n_d_test[i];
            delete[] n_w_test[i];
        }
    }

    delete[] n_w;
    delete[] n_d;

    if (options.perplexity != none) {
        delete[] n_d_test;
        delete[] n_w_test;
        delete[] n_w_dot_test;
    }

    delete[] n_w_dot;
    delete[] n_d_dot;
    for (int i=0; i<B; i++) {
        delete[] psi[i];
        for (int j=0; j<A; j++) {
            delete[] delta_pows[i][j];
        }
        delete[] delta_pows[i];
        delete[] delta_pow_sums[i];
    }
    delete[] psi;

    for (int i=0; i<V; i++) {
        delete[] n_t[i];
    }

    delete[] n_t;

    for (int i=0; i<V && options.perplexity != none; i++) {
        delete[] n_t_test[i];
    }

    if (options.perplexity != none) {
        delete[] n_t_test;
    }

    delete[] delta_pow_sums;
    delete[] delta_pows;
    delete[] corpus_t;
    delete[] hidden;
    delete[] pr;
    if (options.perplexity != none) {
        delete[] pr_test;
    }
    delete[] norm;
    gsl_rng_free (RANDOM_NUMBER);
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------