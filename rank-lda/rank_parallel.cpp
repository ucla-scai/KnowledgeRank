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
#include <tuple>
#include "rank_parallel.h"
#include "rank.h"
#include <algorithm>
#include <stdio.h>
#include <time.h>
//----------------------------------------------------------------------------------
using namespace std;
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int RankParallel::Pop_sample_test(int word, int doc) {
    test = true;
    this->word = word;
    this->doc = doc;
    for (int i=0; i<P; i++) {
        run[i] = true;
    }
    for (int i=0; i<P; i++) {
        while (!populated[i]) {}
        populated[i] = false;
    }

    for (int i=1; i<P; i++) {
        pr_test[ends[i]] += pr_test[ends[i-1]];
    }

    int end_bucket = 0;
    double scale = pr_test[visible_topics.size()-1] * gsl_rng_uniform(RANDOM_NUMBER);

    if (pr_test[ends[0]] <= scale) {
        int low = 0;
        int high = P-1;
        while (low <= high) {
            if (low == high - 1) { end_bucket = high; break; }
            int mid = (low + high) / 2;
            if (pr_test[ends[mid]] > scale) high = mid;
            else low = mid;
        }
    }

    int start = end_bucket == 0 ? 0 : ends[end_bucket-1]+1;
    int end = ends[end_bucket];
    double add = end_bucket == 0 ? 0 : pr_test[ends[end_bucket - 1]];

    int topic = start;
    if (pr_test[start] + add <= scale) {
        int low = start;
        int high = end;
        while (low <= high) {
            if (low == high - 1) { topic = high; break; }
            int mid = (low + high) / 2;
            if (add + pr_test[mid] > scale) high = mid;
            else low = mid;
        }
    }

    return topic;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int RankParallel::Pop_sample(int word, int doc) {
    test = false;
    this->word = word;
    this->doc = doc;
    for (int i=0; i<P; i++) {
        run[i] = true;
    }
    for (int i=0; i<P; i++) {
        while (!populated[i]) {}
        populated[i] = false;
    }

    for (int i=1; i<P; i++) {
        pr[ends[i]] += pr[ends[i-1]];
    }

    int end_bucket = 0;
    double scale = pr[visible_topics.size()-1] * gsl_rng_uniform(RANDOM_NUMBER);

    if (pr[ends[0]] <= scale) {
        int low = 0;
        int high = P-1;
        while (low <= high) {
            if (low == high - 1) { end_bucket = high; break; }
            int mid = (low + high) / 2;
            if (pr[ends[mid]] > scale) high = mid;
            else low = mid;
        }
    }

    int start = end_bucket == 0 ? 0 : ends[end_bucket-1]+1;
    int end = ends[end_bucket];
    double add = end_bucket == 0 ? 0 : pr[ends[end_bucket - 1]];

    int topic = start;
    if (pr[start] + add <= scale) {
        int low = start;
        int high = end;
        while (low <= high) {
            if (low == high - 1) { topic = high; break; }
            int mid = (low + high) / 2;
            if (add + pr[mid] > scale) high = mid;
            else low = mid;
        }
    }

    return topic;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void RankParallel::Hide_topic(int t) {
    Rank::Hide_topic(t);
    ends.clear();
    int gap = visible_topics.size() / P;
    for (int i=0; i<P-1; i++) {
        ends.push_back(((i+1)*gap) - 1);
    }

    ends.push_back(visible_topics.size()-1);
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void RankParallel::gibbs() {

    pthread_t threads[P];
    pthread_attr_t attr;
    void *status;

    int gap = T / P;
    for (int i=0; i<P-1; i++) {
        ends.push_back(((i+1)*gap) - 1);
    }

    ends.push_back(T-1);

    run = new atomic<bool>[P];
    populated = new atomic<bool>[P];

    for (int i=0; i<P; i++) {
        run[i] = false;
        populated[i] = false;
    }

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(int i=0; i < P; i++ ){
        pair<RankParallel *, int>* args = new pair<RankParallel *, int>(this, i);
        pthread_create(&threads[i], NULL, Populate_prob_thread_wrapper, (void*)args);
    }

    Rank::gibbs();
    end = true;

    for (int i=0; i<P; i++) {
        run[i] = true;
    }

    pthread_attr_destroy(&attr);
    for(int i=0; i < P; i++ ){
        pthread_join(threads[i], &status);
    }

    delete[] run;
    delete[] populated;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void RankParallel::Populate_prob_thread(int id) {

    while (!this->end) {
        while (!run[id]) {}
        run[id] = false;

        int start = id == 0 ? 0 : ends[id-1]+1;
        int end = ends[id];

        for (int i=start; i<=end; i++) {
            int t = visible_topics[i];
            if (test) {
                Populate_prob_test(i, t, word, doc, start);
            }
            else {
                Populate_prob(i, t, word, doc, start);
            }
        }
        populated[id] = true;
    }
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void*RankParallel::Populate_prob_thread_wrapper(void *args) {
    pair<RankParallel *, int>* t = (pair<RankParallel *, int>*)args;
    t->first->Populate_prob_thread(t->second);
    delete t;
    pthread_exit(NULL);
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
RankParallel::RankParallel(RankOptions options) : Rank(options) {
    end = false;
    test = false;
}
//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------