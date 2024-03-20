#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iostream>
#include <map>
#include <limits>
#include <boost/math/distributions.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a' literal

typedef std::map< int, std::vector<int> > inner_map;

class Sampler {
public:
    Sampler(std::vector<int> &int_args,
            py::array_t<int> subpop_,
            py::array_t<double> probbase_,
            py::array_t<int> probbase_order_,
            py::dict probbase_level_,
            // std::vector<double> level_values_,
            py::array_t<double> level_values_,
            py::array_t<int> & count_m_,
            py::array_t<int> & count_m_all_,
            py::array_t<int> & count_c_);
    void fit(py::array_t<double> prior_a, double prior_b,
             double jumprange, double trunc_min,
             double trunc_max,
             const py::array_t<double> indic,
             bool contains_missing, int N_gibbs, int burn,
             int thin, py::array_t<double> mu, double sigma2,
             bool use_probbase, bool is_added,
             const py::array_t<double> mu_continue,
             const py::array_t<double> sigma2_continue,
             const py::array_t<double> theta_continue,
             const py::array_t<int> impossible,        // IN PYTHON THESE NEED dtype=np.int32!!!!
             py::array_t<double> & probbase_gibbs_,    // OK to change the rest of these directly
             py::array_t<double> & pnb_mean_,
             py::array_t<double> & levels_gibbs_,      // (Python just passes empty numpy arrays
             py::array_t<double> & p_gibbs_,
             std::vector<int> &n_accept,
             py::array_t<double> & mu_now_,
             py::array_t<double> & sigma2_now_,
             py::array_t<double> & theta_now_,
             py::array_t<double> & p_now_,
             py::array_t<int> & zero_matrix_,          // IN PYTHON THESE NEED dtype=np.int32!!!!
             py::array_t<int> & zero_group_matrix_,    // IN PYTHON THESE NEED dtype=np.int32!!!!
             py::array_t<int> & remove_causes_,
             py::array_t<double> & pnb_,
             py::array_t<double> & y_new_,
             py::array_t<double> & y_,
             py::array_t<double> & parameters_,
             py::dict gui_ctrl,
             bool is_openva_app,
             py::object openva_app);
    int get_n() {return N;};
    int get_s() {return S;};
    int get_c() {return C;};
    int get_n_sub() {return N_sub;};
    int get_pool() {return pool;};
    double runif();
    double rnorm(double mu, double std_dev);
    double rgamma(double alpha, double beta);
private:
    int N;
    int S;
    int C;
    int N_sub;
    int N_level;
    int pool;
    int seed;
    std::mt19937 rng;
    py::array_t<int> subpop;
    py::buffer_info buf_subpop;
    py::array_t<double> probbase;
    py::buffer_info buf_probbase;
    py::array_t<double> new_probbase;   // needed by trunc_beta*
    py::array_t<int> probbase_order;
    py::buffer_info buf_probbase_order;
    py::buffer_info buf_count_m;        // [S][C]
    py::buffer_info buf_count_m_all;    // [S][C]
    py::buffer_info buf_count_c;        // [C]

    // std::vector<double> level_values;
    py::array_t<double> level_values;
    py::buffer_info buf_level_values;
    // std::map<int, std::map<int, std::vector<int> > > probbase_level;
    py::dict probbase_level;

    // void levelize();
    void fill_pnb(bool contains_missing,
                  double *ptr_indic, int *ptr_subpop,
                  py::array_t<double> & pnb_,
                  py::array_t<double> & csmf_sub_,
                  py::array_t<int> & zero_matrix_);
    void sample_y(py::array_t<double> & pnb_,
                  py::array_t<double> & y_);
    void theta_block_update(py::array_t<double> & theta_now_,
                            double jumprange_,
                            py::array_t<double> & mu_now_,
                            py::array_t<double> & sigma2_now_,
                            double *ptr_theta_prev,
                            py::array_t<double> & y_,
                            bool jump_prop_,
                            py::array_t<int> & zero_group_matrix_,
                            int sub_);
    void count_current(double *ptr_indic, py::array_t<double> & y_new_);
    double sample_trunc_beta(double a, double b, double min, double max);
    void trunc_beta2(py::array_t<double> prior_a, double prior_b,
                     double trunc_min, double trunc_max);
    void trunc_beta(py::array_t<double> prior_a, double prior_b,
                    double trunc_min, double trunc_max);
    void trunc_beta_pool(py::array_t<double> prior_a, double prior_b,
                         double trunc_min, double trunc_max);
};

Sampler::Sampler(std::vector<int> &int_args, py::array_t<int> subpop_,
                 py::array_t<double> probbase_, py::array_t<int> probbase_order_,
                 py::dict probbase_level_, py::array_t<double> level_values_,
                 py::array_t<int> & count_m_, py::array_t<int> & count_m_all_,
                 py::array_t<int> & count_c_)
    // : N{int_args[0], S{int_args[1]}, C{int_args[2]}, N_sub{int_args[3]},
    //  N_level{int_args[4]}, pool{int_args[5]}
{
    if (int_args.size() != 7) {
        throw std::runtime_error("int_args needs 7 elements (N, S, C, N_sub, N_level, pool, seed)");
    }
    N = int_args[0];  // just passed as a Python list; should these be in initializer list?
    S = int_args[1];
    C = int_args[2];
    N_sub = int_args[3];
    N_level = int_args[4];
    pool = int_args[5];
    seed = int_args[6];
    std::mt19937 rng(seed);

    subpop = subpop_;
    buf_subpop = subpop.request();
    if (buf_subpop.shape[0] != N)
        throw std::runtime_error("subpop needs to have size N");

    py::buffer_info buf_pb_ = probbase_.request();
    probbase = py::array_t<double>(buf_pb_);   // make copy of probbase so we can make changes
    buf_probbase = probbase.request();
    if (buf_probbase.shape[0] != S && buf_probbase.shape[1] != C)
        throw std::runtime_error("probbase needs to have shape (S, C)");
    // needed by trunc_beta family of functions
    new_probbase = py::array_t<double>(buf_pb_);   // used to update probbase in trunc_beta*

    py::buffer_info buf_pbo_ = probbase_order_.request();
    probbase_order = py::array_t<int>(buf_pbo_);
    buf_probbase_order = probbase_order.request();
    if (buf_probbase_order.shape[0] != S && buf_probbase_order.shape[0] != C)
        throw std::runtime_error("probbase_order needs to have shape (S, C)");

    probbase_level = probbase_level_;

    level_values = level_values_;
    buf_level_values = level_values.request();
    // if (level_values.size() != 15)
    if (buf_level_values.shape[0] != 15)
        throw std::runtime_error("level_values needs to have 15 elements");

    buf_count_m = count_m_.request();
    if (buf_count_m.shape[0] != S && buf_count_m.shape[1] != C)
        throw std::runtime_error("count_m needs to have shape (S, C)");
    buf_count_m_all = count_m_all_.request();
    if (buf_count_m_all.shape[0] != S && buf_count_m_all.shape[1] != C)
        throw std::runtime_error("count_m_all needs to have shape (S, C)");
    buf_count_c = count_c_.request();
    if (buf_count_c.shape[0] != C)
        throw std::runtime_error("count_c needs to have length C");

    // levelize();
}

double Sampler::runif(){
    std::uniform_real_distribution<double> rng_u{0.0, 1.0};
    return rng_u(rng);
}

double Sampler::rnorm(double mu, double std_dev) {
    std::normal_distribution<double> rng_n{mu, std_dev};
    return rng_n(rng);
}

double Sampler::rgamma(double alpha, double beta) {
    std::gamma_distribution<double> rng_g{alpha, beta};
    return rng_g(rng);
}

// void Sampler::levelize()
// {
//     typedef std::map< int, std::vector<int> > inner_map;
//     // typedef std::map< int, inner_map > hash_map;
//     int *ptr = (int *) buf_probbase_order.ptr;

//     if (pool < 2) {
//      for (int s = 0; s < S; ++s) {
//          for (int c = 0; c < C; ++ c) {
//              // get the level of the s-c combination
//              int level = (int) ptr[s*C + c];
//              // initialize if this cause if needed
//              if (probbase_level.find(c) == probbase_level.end()) {
//                  probbase_level.insert(
//                      std::pair<int, inner_map>(c, inner_map()) );
//              }
//              if (probbase_level[c].find(level) == probbase_level[c].end()) {
//                  probbase_level[c].insert(
//                      std::pair<int, std::vector<int> >(level, std::vector<int>()));
//              }
//              probbase_level[c][level].push_back(s);
//          }
//      }
//     } else {
//      for (int s = 0; s < S; ++s) {
//          for (int c = 0; c < C; ++ c) {
//              // get the level of the s-c combination
//              int level = (int) ptr[s*C + c];
//              // initialize if this cause if needed
//              if (probbase_level.find(s) == probbase_level.end()) {
//                  probbase_level.insert(
//                      std::pair<int, inner_map>(s, inner_map()) );
//              }
//              if (probbase_level[s].find(level) == probbase_level[s].end()) {
//                  probbase_level[s].insert(
//                      std::pair<int, std::vector<int> >(level, std::vector<int>()));
//              }
//              probbase_level[s][level].push_back(c);
//          }
//      }
//     }
// }

void Sampler::fill_pnb(bool contains_missing,
                       double *ptr_indic, int *ptr_subpop,
                       py::array_t<double> & pnb_,
                       py::array_t<double> & csmf_sub_,
                       py::array_t<int> & zero_matrix_){
    
    // create pointers to get access to Numpy arrays
    auto pnb = pnb_.mutable_unchecked<2>();
    auto zero_matrix = zero_matrix_.mutable_unchecked<2>();
    auto csmf_sub = csmf_sub_.mutable_unchecked<2>();
    double *ptr_probbase = (double *) buf_probbase.ptr;
    // initialize p.nb matrix to p.hat
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            int sub = ptr_subpop[n];
            pnb(n, c) = csmf_sub(sub, c) * zero_matrix(n, c);
        }
    }
    // calculate posterior
    for (int n = 0; n < N; ++n) {
        // find which symptoms are not missing for this death
        double row_total = 0.0;
        std::vector<int> nomissing;
        for (int s = 0; s < S; ++s) { // this needs indic[n].length?
            if (ptr_indic[n*S + s] >= 0) nomissing.push_back(s);
        }
        // loop over cause-symptom combination to calculate naive bayes prob
        // adding in normalization step
        for (int c = 0; c < C; ++c) {
            for (int s : nomissing) {
                double pb_val = ptr_probbase[s*C + c];
                pnb(n, c) *= (ptr_indic[n*S + s] > 0) ? pb_val: (1 - pb_val);
            }
            row_total += pnb(n, c);
        }
        for (int c = 0; c < C; ++c) {
            if (row_total == 0.0) {
                pnb(n, c) = 0.0;
            } else {
                pnb(n, c) = pnb(n, c) / row_total;
            }
        }
    }
}

void Sampler::sample_y(py::array_t<double> & pnb_,
                       py::array_t<double> & y_new_) {
    
    auto pnb = pnb_.mutable_unchecked<2>();
    auto y_new = y_new_.mutable_unchecked<1>();
    //loop over every death
    for (int n = 0; n < (int) pnb.shape(0); ++n){
        double u = runif();
        double cum = 0.0;
        for (int c = 0; c < (int) pnb.shape(1); ++c) {
            cum += pnb(n, c);
            if (u < cum) {
                y_new(n) = c;
                break;
            }
        }
    }
}

void Sampler::theta_block_update(py::array_t<double> & theta_now_,
                                 double jumprange_,
                                 py::array_t<double> & mu_now_,
                                 py::array_t<double> & sigma2_now_,
                                 double *ptr_theta_prev,
                                 py::array_t<double> & y_,
                                 bool jump_prop_,
                                 py::array_t<int> & zero_group_matrix_,
                                 int sub_) {

    // store new values then replace with theta_prev if rejected
    // (in accept/reject step)
    auto theta_new = theta_now_.mutable_unchecked<2>();
    auto mu_now = mu_now_.mutable_unchecked<2>();
    auto sigma2_now = sigma2_now_.mutable_unchecked<1>();
    auto y = y_.mutable_unchecked<2>();
    auto zero_group_matrix = zero_group_matrix_.mutable_unchecked<2>();
    int sub = sub_;

    if (jump_prop_) {
        // not using
    }
    // initialize new theta
    int fix = 0;
    // in Java version only a row is passed to this function (i.e., zero_group_matrix[sub])
    // so we want to iterate over the columns of zero_group_matrix
    for (int i = 0; i < (int) zero_group_matrix.shape(1); ++i) {
        if (zero_group_matrix(sub, i) != 0) {
            fix = i;
            break;
        }
    }
    theta_new(sub, fix) = 1.0;
    // calculate sum(exp(theta)), the normalizing constant
    double expsum = std::exp(1.0);
    double expsum_new = std::exp(1.0);

    //sample new theta proposal
    for (int c = fix + 1; c < C; ++c) {
        int sub_idx = sub * C + c;
        theta_new(sub, c) = rnorm(ptr_theta_prev[sub_idx],
                                  jumprange_) * zero_group_matrix(sub, c);
        expsum += std::exp(ptr_theta_prev[sub_idx]);
        expsum_new += std::exp(theta_new(sub, c));
    }
    double log_trans = 0.0;
    for (int c = 0; c < C; ++c) {
        int sub_idx = sub * C + c;
        if (zero_group_matrix(sub, c) > 0) {
            double term1 = (theta_new(sub, c) - mu_now(sub, c)) *
                (theta_new(sub, c) - mu_now(sub, c));
            double term2 = (ptr_theta_prev[sub_idx] - mu_now(sub, c)) *
                (ptr_theta_prev[sub_idx] - mu_now(sub, c));
            double diffquad = term1 - term2;
            log_trans += y(sub, c) *
                (theta_new(sub, c) - ptr_theta_prev[sub_idx] - std::log(expsum_new / expsum))
                - 1.0 / (2.0*sigma2_now(sub)) * diffquad;
        }

    }
    // accept or reject
    double u = std::log(runif());
    if (log_trans >= u) {
        // accept: theta_new already stored in theta_now (above)
        // py::print("+");
    } else {
      // reject: put theta_prev back in to theta_now
      // py::print("-");
        for (int c = 0; c < C; ++c) {
            int sub_idx = sub * C + c;
            theta_new(sub, c) = ptr_theta_prev[sub_idx];
        }
    }
}

void Sampler::count_current(double *ptr_indic,
                            py::array_t<double> & y_new_) {

    int *ptr_count_m = (int *) buf_count_m.ptr;            // counts of yes (S*C)
    int *ptr_count_m_all = (int *) buf_count_m_all.ptr;    // counts of yes and no (S*C)
    int *ptr_count_c = (int *) buf_count_c.ptr;            // counts of causes (C)
    auto y_new = y_new_.mutable_unchecked<1>();

    for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
            int idx = s*C + c;
            if (s == 0) ptr_count_c[c] = 0;
            ptr_count_m[idx] = 0;
            ptr_count_m_all[idx] = 0;
        }
    }

    // loop over all inidviduals
    for (int n = 0; n < N; ++n) {
        // cause of this death
        int c_current = (int) y_new(n);
        // add to toal counts
        ptr_count_c[c_current] += 1;
        // loop over all symptoms
        for (int s = 0; s < S; ++s) {
            // add to counts of this symptom's appearance
            int idx = s*C + c_current;
            if (ptr_indic[n*S + s] == 1) {
                ptr_count_m[idx] += 1;
                ptr_count_m_all[idx] += 1;
            } else if (ptr_indic[n*S + s] == 0) {
                ptr_count_m_all[idx] += 1;
            }
        }
    }
}

double Sampler::sample_trunc_beta(double a_, double b_, double min_, double max_) {
    boost::math::beta_distribution<> beta_dist(a_, b_);
    double value = min_;
    double ymin = cdf(beta_dist, min_);
    double ymax = cdf(beta_dist, max_);
    // handling boundary case
    if (std::abs(ymax - ymin) < 1e-8) {
        return (max_ + min_) / 2.0;
    }
    double x = runif() * (ymax - ymin) + ymin;
    value = quantile(beta_dist, x);
    return value;
}

void Sampler::trunc_beta2(py::array_t<double> prior_a, double prior_b,
                          double trunc_min, double trunc_max) {

    // typedef std::map< int, std::vector<int> > inner_map;

    auto proxy_pbase = probbase.mutable_unchecked<2>();
    auto proxy_new_pbase = new_probbase.mutable_unchecked<2>();
    auto proxy_prior_a = prior_a.mutable_unchecked<1>();
    for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
            proxy_new_pbase(s, c) = proxy_pbase(s, c);
        }
    }
    int *ptr_count_m = (int *) buf_count_m.ptr;            // counts of yes (S*C)
    int *ptr_count_m_all = (int *) buf_count_m_all.ptr;    // counts of yes and no (S*C)
    double a = 0.0;
    double b = 0.0;
    // loop over symptoms 
    for (int s = 0; s < S; ++s) {
        // find which level-symptom combinations under this cause
        // inner_map levels_under_s = probbase_level[s];
        py::dict levels_under_s = probbase_level[py::cast(s)];
        // find the list of all levels present under cause c

        std::vector<int> exist_levels_under_s;
        for (int l = 1; l <= N_level; ++l) {
            // if (levels_under_s.find(l) != levels_under_s.end()) {
            if (levels_under_s.contains(py::cast(l))){
                exist_levels_under_s.push_back(l);
            }
        }
        // loop over level l, in ascending order
        for (int index = 0; index < (int) exist_levels_under_s.size(); ++index) {
            int l_current = exist_levels_under_s[index];
        // loop over symptoms s in the level l
            // for (int c : levels_under_s[l_current]) {
            py::list levels_under_s_l = levels_under_s[py::cast(l_current)];
            for (auto c : levels_under_s_l) {
                int count = ptr_count_m[s*C + c.cast<int>()];
                int count_all = ptr_count_m_all[s*C + c.cast<int>()];
                double lower = 0.0;
                double upper = 1.0;
                if (index == 0) {
                    // find which level is next
                    int l_next = exist_levels_under_s[index + 1];
                    // find the max of those symptoms
                    lower = std::numeric_limits<double>::min();
                    py::list levels_under_s_l_next = levels_under_s[py::cast(l_next)];
                    // for (int i : levels_under_s[l_next]) {
                    for (auto i : levels_under_s_l_next) {
                        if (proxy_pbase(s, i.cast<int>()) > lower) lower = proxy_pbase(s, i.cast<int>());
                    }
                    // make sure not lower than lower bound
                    lower = std::max(lower, trunc_min);
                    upper = trunc_max;
                    // if the lower level
                } else if (index == (int) exist_levels_under_s.size() - 1) {
                    lower = trunc_min;
                    int l_prev = exist_levels_under_s[index - 1];
                    upper = std::numeric_limits<double>::max();
                    py::list levels_under_s_l_prev = levels_under_s[py::cast(l_prev)];
                    // for (int i : levels_under_s[l_prev]) {
                    for (auto i : levels_under_s_l_prev) {
                        if (proxy_new_pbase(s, i.cast<int>()) < upper) upper = proxy_new_pbase(s, i.cast<int>());
                    }
                    upper = std::min(upper, trunc_max);
                    // if in the middle
                } else {
                    int l_next = exist_levels_under_s[index + 1];
                    lower = std::numeric_limits<double>::min();
                    // for (int i : levels_under_s[l_next]) {
                    py::list levels_under_s_l_next = levels_under_s[py::cast(l_next)];
                    for (auto i : levels_under_s_l_next) {
                        if (proxy_pbase(s, i.cast<int>()) > lower) lower = proxy_pbase(s, i.cast<int>());
                    }
                    lower = std::max(lower, trunc_min);
                    int l_prev = exist_levels_under_s[index - 1];
                    upper = std::numeric_limits<double>::max();
                    // for (int i : levels_under_s[l_prev]) {
                    py::list levels_under_s_l_prev = levels_under_s[py::cast(l_prev)];
                    for (auto i : levels_under_s_l_prev) {
                        if (proxy_new_pbase(s, i.cast<int>()) < upper) upper = proxy_new_pbase(s, i.cast<int>());
                    }
                    upper = std::min(upper, trunc_max);
                }
                // if range is invalide, use higher case
                if (lower >= upper) {
                    proxy_new_pbase(s, c.cast<int>()) = upper;
                } else {
                    // find the beta distribution parameters, note level starting from 1
                    a = proxy_prior_a(l_current-1) + count;
                    b = prior_b + count_all - a;
                    proxy_new_pbase(s, c.cast<int>()) = sample_trunc_beta(a, b, lower, upper);
                }
            }
        }
        // update this column of probbase
        for (int c = 0; c < C; ++c) {
            proxy_pbase(s, c) = proxy_new_pbase(s, c);
        }
    }
}

void Sampler::trunc_beta(py::array_t<double> prior_a, double prior_b,
                         double trunc_min, double trunc_max) {

    // typedef std::map< int, std::vector<int> > inner_map;

    auto proxy_pbase = probbase.mutable_unchecked<2>();
    auto proxy_new_pbase = new_probbase.mutable_unchecked<2>();
    auto proxy_prior_a = prior_a.mutable_unchecked<1>();
    for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
            proxy_new_pbase(s, c) = proxy_pbase(s, c);
        }
    }
    int *ptr_count_m = (int *) buf_count_m.ptr;            // counts of yes (S*C)
    int *ptr_count_m_all = (int *) buf_count_m_all.ptr;    // counts of yes and no (S*C)
    double a = 0.0;
    double b = 0.0;
    // loop over causes c
    for (int c = 0; c < C; ++c) {
        // find which level-symptom combinations under this cause
        // inner_map levels_under_c = probbase_level[c];
        py::dict levels_under_c = probbase_level[py::cast(c)];
        //find the list of all levels present under cause c
        std::vector<int> exist_levels_under_c;
        for (int l = 1; l <= N_level; ++l) {
            // if (levels_under_c.find(l) != levels_under_c.end()) {
            if (levels_under_c.contains(py::cast(l))) {
                exist_levels_under_c.push_back(l);
            }
        }
        // loop over level l, in ascending order
        for (int index = 0; index < (int) exist_levels_under_c.size(); ++index) {
            int l_current = exist_levels_under_c[index];
        // loop over symptoms s in the level l
            // for (int s : levels_under_c[l_current]) {
            py::list levels_under_c_l_current = levels_under_c[py::cast(l_current)];
            for (auto s : levels_under_c_l_current) {
                int count = ptr_count_m[s.cast<int>()*C + c];
                int count_all = ptr_count_m_all[s.cast<int>()*C + c];
                double lower = 0.0;
                double upper = 1.0;
                if (index == 0) {
                    // find which level is next
                    int l_next = exist_levels_under_c[index + 1];
                    // find the max of those symptoms
                    lower = std::numeric_limits<double>::min();
                    // for (int i : levels_under_c[l_next]) {
                    py::list levels_under_c_l_next = levels_under_c[py::cast(l_next)];
                    for (auto i : levels_under_c_l_next) {
                        if (proxy_pbase(i.cast<int>(), c) > lower) lower = proxy_pbase(i.cast<int>(), c);
                    }
                    // make sure not lower than lower bound
                    lower = std::max(lower, trunc_min);
                    upper = trunc_max;
                    // if the lower level
                } else if (index == (int) (exist_levels_under_c.size() - 1)) {
                    lower = trunc_min;
                    int l_prev = exist_levels_under_c[index - 1];
                    upper = std::numeric_limits<double>::max();
                    // for (int i : levels_under_c[l_prev]) {
                    py::list levels_under_c_l_prev = levels_under_c[py::cast(l_prev)];
                    for (auto i : levels_under_c_l_prev) {
                        if (proxy_new_pbase(i.cast<int>(), c) < upper) upper = proxy_new_pbase(i.cast<int>(), c);
                    }
                    upper = std::min(upper, trunc_max);
                    // if in the middle
                } else {
                    int l_next = exist_levels_under_c[index + 1];
                    lower = std::numeric_limits<double>::min();
                    // for (int i : levels_under_c[l_next]) {
                    py::list levels_under_c_l_next = levels_under_c[py::cast(l_next)];
                    for (auto i : levels_under_c_l_next) {
                        if (proxy_pbase(i.cast<int>(), c) > lower) lower = proxy_pbase(i.cast<int>(), c);
                    }
                    lower = std::max(lower, trunc_min);
                    int l_prev = exist_levels_under_c[index - 1];
                    upper = std::numeric_limits<double>::max();
                    // for (int i : levels_under_c[l_prev]) {
                    py::list levels_under_c_l_prev = levels_under_c[py::cast(l_prev)];
                    for (auto i : levels_under_c_l_prev) {
                        if (proxy_new_pbase(i.cast<int>(), c) < upper) upper = proxy_new_pbase(i.cast<int>(), c);
                    }
                    upper = std::min(upper, trunc_max);
                }
                // if range is invalide, use higher case
                if (lower >= upper) {
                    proxy_new_pbase(s.cast<int>(), c) = upper;
                } else {
                    // find the beta distribution parameters, note level starting from 1
                    a = proxy_prior_a(l_current-1) + count;
                    b = prior_b + count_all - a;
                    proxy_new_pbase(s.cast<int>(), c) = sample_trunc_beta(a, b, lower, upper);
                }
            }
        }
        // update this column of probbase
        for (int s = 0; s < S; ++s) {
            proxy_pbase(s, c) = proxy_new_pbase(s, c);
        }
    }
}

void Sampler::trunc_beta_pool(py::array_t<double> prior_a, double prior_b,
                              double trunc_min, double trunc_max) {

//    typedef std::map< int, std::vector<int> > inner_map;

    auto proxy_pbase = probbase.mutable_unchecked<2>();
    auto proxy_pbase_order = probbase_order.mutable_unchecked<2>();
    auto proxy_prior_a = prior_a.mutable_unchecked<1>();
    double *ptr_level_values = (double *) buf_level_values.ptr;
    int *ptr_count_m = (int *) buf_count_m.ptr;            // counts of yes (S*C)
    int *ptr_count_m_all = (int *) buf_count_m_all.ptr;    // counts of yes and no (S*C)
    double a = 0.0;
    double b = 0.0;
    // assume all levels exist in data, if not, need to modify Python codes before calling
    std::vector<double> new_level_values(N_level, 0.0);
    for (int l = 1; l < N_level; ++l) {
        int count = 0;
        int count_all = 0;
        //count appearances
        for (int c = 0; c < C; ++c) {
            // if (probbase_level[c].find(l) != probbase_level[c].end()) {
            py::dict level_c = probbase_level[py::cast(c)];
            if (level_c.contains(py::cast(l))) {
                // for (int s : probbase_level[c][l]) {
                py::list level_c_l = level_c[py::cast(l)];
                // for (int s : probbase_level[c][l]) {
                for (auto s : level_c_l) {
                    count += ptr_count_m[s.cast<int>()*C + c];
                    count_all += ptr_count_m_all[s.cast<int>()*C + c];
                }
            }
        }
        double lower = 0;
        double upper = 1;
        // note l starts from 1, level_values have index starting from 0
        if (l == 1) {
            // lower bound is the max of this next level
            // lower = std::max(level_values[l], trunc_min);
            lower = std::max(ptr_level_values[l], trunc_min);
            upper = trunc_max;
        } else if (l == N_level) {
            lower = trunc_min;
            // upper bound is the min of previous level
            upper = std::min(new_level_values[l-2], trunc_max);
        } else {
            // lower = std::max(level_values[l], trunc_min);
            lower = std::max(ptr_level_values[l], trunc_min);
            upper = std::min(new_level_values[l-2], trunc_max);
        }
        if (lower >= upper) {
            new_level_values[l-1] = upper;
        } else {
            a = proxy_prior_a(l-1) + count;
            b = prior_b + count_all - a;
            new_level_values[l-1] = sample_trunc_beta(a, b, lower, upper);
        }
    }
    // level_values = new_level_values;
    for (int l = 0; l < N_level; ++l) {
        ptr_level_values[l] = new_level_values[l];
    }
    for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
            int idx = (int) (proxy_pbase_order(s, c) - 1);
            proxy_pbase(s, c) = ptr_level_values[idx];
        }
    }
}

void Sampler::fit(py::array_t<double> prior_a, double prior_b,
                  double jumprange, double trunc_min,
                  // double trunc_max, const py::array_t<int> indic,  // assumes numpy.int32
                  double trunc_max, const py::array_t<double> indic,  // assumes numpy.int32
                  bool contains_missing, int N_gibbs, int burn,
                  int thin, py::array_t<double> mu, double sigma2,
                  bool use_probbase, bool is_added,
                  const py::array_t<double> mu_continue,
                  const py::array_t<double> sigma2_continue,
                  const py::array_t<double> theta_continue,
                  const py::array_t<int> impossible,
                  py::array_t<double> & probbase_gibbs_,
                  py::array_t<double> & levels_gibbs_,
                  py::array_t<double> & p_gibbs_,
                  py::array_t<double> & pnb_mean_,
                  std::vector<int> &n_accept,
                  py::array_t<double> & mu_now_,
                  py::array_t<double> & sigma2_now_,
                  py::array_t<double> & theta_now_,
                  py::array_t<double> & p_now_,
                  py::array_t<int> & zero_matrix_,
                  py::array_t<int> & zero_group_matrix_,
                  py::array_t<int> & remove_causes_,
                  py::array_t<double> & pnb_,
                  py::array_t<double> & y_new_,
                  py::array_t<double> & y_,
                  py::array_t<double> & parameters_,
                  py::dict gui_ctrl,
                  bool is_openva_app,
                  py::object openva_app)
{
       
    double d;
    int progress = 0;
    int *ptr_subpop = (int *) buf_subpop.ptr;
    py::buffer_info buf_indic = indic.request();  // these are ints, but not treated as such in Python -- need np.int32
    if (buf_indic.shape[0] != N && buf_indic.shape[1] != S)
        throw std::runtime_error("indic needs to have shape (N, S)");
    double *ptr_indic = (double *) buf_indic.ptr;
    py::buffer_info buf_mu = mu.request();
    if (buf_mu.shape[0] != C)
        throw std::runtime_error("mu needs to have shape (C,)");
    double *ptr_mu = (double *) buf_mu.ptr;
    py::buffer_info buf_mu_cont = mu_continue.request();
    if (buf_mu_cont.shape[0] != N_sub && buf_mu_cont.shape[1] != C)
        throw std::runtime_error("mu_continue needs to have shape (N_sub, C)");
    double *ptr_mu_cont = (double *) buf_mu_cont.ptr;
    py::buffer_info buf_sigma2_cont = sigma2_continue.request();
    if (buf_sigma2_cont.shape[0] != N_sub)
        throw std::runtime_error("sigma2_cont needs to have shape (N_sub,)");
    double *ptr_sigma2_cont = (double *) buf_sigma2_cont.ptr;    
    py::buffer_info buf_theta_cont = theta_continue.request();
    if (buf_theta_cont.shape[0] != N_sub && buf_theta_cont.shape[1] != C)
        throw std::runtime_error("theta_continue needs to have shape (N_sub, C)");
    double *ptr_theta_cont = (double *) buf_theta_cont.ptr;
    py::buffer_info buf_impossible = impossible.request();
    // this shape depends arguements for impossible causes
    // if (buf_impossible.shape[0] && buf_theta_cont.shape[1] != C)
    //  throw std::runtime_error("theta_continue needs to have shape (N_sub, C)");
    int *ptr_impossible = (int *) buf_impossible.ptr;
    double *ptr_probbase = (double *) buf_probbase.ptr;
    double *ptr_level_values = (double *) buf_level_values.ptr;

    size_t size_n_sub = buf_mu_cont.shape[0];
    size_t size_c = buf_mu_cont.shape[1];

    if (is_openva_app == false) {
        py::print("InSilicoVA Sampler Initiated, ", N_gibbs, " Iterations to Sample\n");
    }
    int N_thin = (int) ((N_gibbs - burn) / (thin));
    int n_report = std::max(N_gibbs/20, 100);
    if (N_gibbs < 200) n_report = 50;

    auto pnb = pnb_.mutable_unchecked<2>();
    // probbase at each thinned iteration (N_thin, S, C)
    auto probbase_gibbs = probbase_gibbs_.mutable_unchecked<3>();
    // probbase levels at each thinned iterations (N_thin, N_level)
    auto levels_gibbs = levels_gibbs_.mutable_unchecked<2>();
    // csmf at each thinned iteration (N_thin, N_sub, C)
    auto p_gibbs = p_gibbs_.mutable_unchecked<3>();
    // individual probability at each thinned iteration (N, C)
    auto pnb_mean = pnb_mean_.mutable_unchecked<2>();
    auto mu_now = mu_now_.mutable_unchecked<2>();
    auto sigma2_now = sigma2_now_.mutable_unchecked<1>();
    auto theta_now = theta_now_.mutable_unchecked<2>();
    py::buffer_info buf_theta_now = theta_now_.request();
    py::array_t<double> theta_prev;
    py::buffer_info buf_theta_prev;
    auto p_now = p_now_.mutable_unchecked<2>();
    auto parameters = parameters_.mutable_unchecked<1>();
    auto zero_matrix = zero_matrix_.mutable_unchecked<2>();
    auto zero_group_matrix = zero_group_matrix_.mutable_unchecked<2>();
    auto remove_causes = remove_causes_.mutable_unchecked<1>();

    py::array_t<double> y_new = y_new_;
    py::buffer_info buf_y_new = y_new.request();
    double *ptr_y_new = (double *) buf_y_new.ptr;
    auto y = y_.mutable_unchecked<2>();

    if (!is_added) {
        for (size_t sub = 0; sub < size_n_sub; ++sub) {
            for (size_t i = 0; i < size_c; ++i) {
                mu_now(sub, i) = ptr_mu[i];
                theta_now(sub, i) = 1.0;
            }
            sigma2_now(sub) = sigma2;

            double expsum = exp(1.0);
            for (int c = 1; c < C; ++c) {
                d = runif();
                theta_now(sub, c) = log(d * 100);
                expsum += exp(theta_now(sub, c));
            }
            for (int c = 0; c < C; ++c) {
                p_now(sub, c) = exp(theta_now(sub, c)) / expsum;
            }
        }
    } else {
        // py::print("Made it to 2nd run of sampler");
        for (size_t sub = 0; sub < size_n_sub; ++sub) {
            for (size_t i = 0; i < size_c; ++i) {
                mu_now(sub, i) = ptr_mu_cont[sub * size_c + i];
                theta_now(sub, i) = ptr_theta_cont[sub * size_c + i];
            }
            sigma2_now(sub) = ptr_sigma2_cont[sub];
            // py::print("sigma2_now(sub): ", sigma2_now(sub));

            // recalculate p from theta
            double expsum = exp(1.0);
            for (int c = 1; c < C; ++c) {
                expsum += exp(theta_now(sub, c));
            }
            for (int c = 0; c < C; ++c) {
                p_now(sub, c) = exp(theta_now(sub, c)) / expsum;
            }
        }
        // py::print("p_now(0, 0): ", p_now(0, 0));
    }

    // check impossible causes?
    int ncol_impossible = (int) buf_impossible.shape[1];
    int nrow_impossible = (int) buf_impossible.shape[0];
    bool check_impossible = (ncol_impossible == 3);

    for (int i=0; i<N; ++i) {
        for (int j=0; j<C; j++) {
            zero_matrix(i, j) = 1;
        }
    }

    // indic is (N, S)
    if (check_impossible) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < nrow_impossible; ++k) {
                int index = k * ncol_impossible;
                int imp_col0 = ptr_impossible[index];
                int imp_col1 = ptr_impossible[index + 1];
                int imp_col2 = ptr_impossible[index + 2];
                if ((ptr_indic[i*S + imp_col1] == 1) & (imp_col2 == 0)) {
                    zero_matrix(i, imp_col0) = 0;
                }
                if ((ptr_indic[i*S + imp_col1] == 0) & (imp_col2 == 1)) {
                    zero_matrix(i, imp_col0) = 0;
                }
            }
        }
    }
    // check if specific causes are impossible for a whole subpopulation
    if (check_impossible) {
        for (int j = 0; j < C; ++j) {
            for (int i = 0; i < N; ++i) {
                zero_group_matrix(ptr_subpop[i], j) += zero_matrix(i, j);
            }
        }
        for (int j = 0; j < C; ++j) {
            for (int i = 0; i < N_sub; ++i) {
                if (zero_group_matrix(i, j) != 0)
                    zero_group_matrix(i, j) = 1;
                remove_causes(i) += 1 - zero_group_matrix(i, j);
            }
        }
    } else {
        for (int i = 0; i < N_sub; ++i) {
            for (int j = 0; j < C; ++j) {
                zero_group_matrix(i, j) = 1;
            }
        }       
    }

    // reinitiate after checking impossible
    if (!is_added) {
        for (int sub = 0; sub < N_sub; ++sub) {
            int fix = 0;
            for (int c = 1; c < C; ++c) {
                if (zero_group_matrix(sub, c) > 0) {
                    fix = c;
                    break;
                }
            }
            theta_now(sub, fix) = 1.0;
            double expsum = exp(1.0);
            for (int c = (fix + 1); c < C; ++c) {
                d = runif() * 100;
                theta_now(sub, c) = log(d);
                expsum += exp(theta_now(sub, c)) * zero_group_matrix(sub, c);
            }
            for (int c = 0; c < C; ++c) {
                p_now(sub, c) = exp(theta_now(sub, c)) *
                    zero_group_matrix(sub, c) / expsum;
            }
        }
    }

    // first time pnb (naive bayes probability) calculation
    fill_pnb(true, ptr_indic, ptr_subpop, pnb_, p_now_, zero_matrix_);

    // start loop
    auto start = std::chrono::system_clock::now();
    std::string key = "break";
    bool early_stop = false;
    for (int k = 0; k < N_gibbs; ++k) {
        early_stop = gui_ctrl[py::cast(key)].cast<bool>();
        if (early_stop) break;

        // sample new y vector
        sample_y(pnb_, y_new_);

        // count the appearance of each cause
        for (int sub = 0; sub < N_sub; ++sub) {
            for (int c = 0; c < C; ++c) {
                y(sub, c) = 0;
            }
        }
        for (int n = 0; n < N; ++n) {
            y(ptr_subpop[n], ptr_y_new[n]) += 1;
        }

        for (int sub = 0; sub < N_sub; ++sub) {
            //sample mu
            double mu_mean = 0.0;
            for (int c = 0; c < C; ++c) {
                mu_mean += theta_now(sub, c);
            }
            mu_mean =  mu_mean / (C - remove_causes(sub) + 0.0);
            mu_mean = rnorm(mu_mean, std::sqrt(sigma2_now(sub) /
                                               (C - remove_causes(sub) + 0.0)));
            for (int c = 0; c < C; ++c) {
                mu_now(sub, c) = mu_mean;
            }
            //sample sigma2
            double shape = (C - remove_causes(sub) - 1.0) / 2.0;
            double rate2 = 0.0;
            for (int c = 0; c < C; ++c) {
                rate2 += std::pow(theta_now(sub, c) - mu_now(sub, c) *
                                  zero_group_matrix(sub, c), 2);
            }
            sigma2_now(sub) = 1.0 / rgamma(shape, 2.0 / rate2);
            //sample theta
            theta_prev = py::array_t<double>(buf_theta_now);
            buf_theta_prev = theta_prev.request();
            double *ptr_theta_prev = (double *) buf_theta_prev.ptr;
            theta_block_update(theta_now_, jumprange, mu_now_, sigma2_now_,
                               ptr_theta_prev, y_, false, zero_group_matrix_,
                               sub);

            for (int j = 0; j < C; ++j) {
                if (theta_now(sub, j) != ptr_theta_prev[sub*C + j]) {
                    n_accept[sub] += 1;
                    break;
                }
            }
            // calculate phat
            double expsum = 0.0;
            for (int c = 0; c < C; ++c) {
                expsum += std::exp(theta_now(sub, c) * zero_group_matrix(sub, c));
            }
            for (int c = 0; c < C; ++c) {
                p_now(sub, c) = std::exp(theta_now(sub, c)) *
                    zero_group_matrix(sub, c) / expsum;
            }            
        }

        if (use_probbase) {
            // skip the update step for probbase
        } else {
            count_current(ptr_indic, y_new_);
            if (pool == 0) {
                trunc_beta_pool(prior_a, prior_b, trunc_min, trunc_max);
            } else if (pool == 1) {
                trunc_beta(prior_a, prior_b, trunc_min, trunc_max);
            } else if (pool == 2) {
                trunc_beta2(prior_a, prior_b, trunc_min, trunc_max);
            }
        }

        fill_pnb(true, ptr_indic, ptr_subpop, pnb_, p_now_, zero_matrix_);

        // format output message
        if ((k % 10 == 0) & (is_openva_app == false)) {
            py::print(".", "end"_a="", "flush"_a=true);
        }
        if ( (k % n_report == 0) & (k != 0) & (is_openva_app == false)) {
            std::string message = "\nIteration: " + std::to_string(k) + " \n";
            for (int sub = 0; sub < N_sub; ++sub) {
                double ratio = std::ceil(100.0 * n_accept[sub] / (k + 0.0)) / 100.0;
                std::string sub_message = "Sub-population " +
                    std::to_string(sub) + " acceptance ratio: " +
                    std::to_string(ratio) + '\n';
                message += sub_message;
            }
            py::print(message);
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            double t1 = duration / 1000.0 / 60.0;
            double t2 = t1/(k + 0.0) * (N_gibbs - k);
            t1 = std::ceil(t1 * 100.0) / 100.0;
            t2 = std::ceil(t2 * 100.0) / 100.0;
            
            py::print(t1, "min elapsed, ", t2, "min remaining", '\n');
        }
        if ((k % 10 == 0) & is_openva_app) {
            progress = 100 * (k + 1) / N_gibbs;
            openva_app.attr("emit")(progress);
        }

        // note this condition includes the first iteration
        if ( (k >= burn) & ((k - burn + 1) % thin == 0) ) {
            int save = ((int) ((k - burn + 1)/thin)) - 1;
            for (int d1 = 0; d1 < N; ++d1) {
                for (int d2 = 0; d2 < C; ++d2) {
                    pnb_mean(d1, d2) += pnb(d1, d2);
                }
            }
            for (int d1 = 0; d1 < N_sub; ++d1) {
                for (int d2 = 0; d2 < C; ++d2) {
                    p_gibbs(save, d1, d2) = p_now(d1, d2);
                }
            }
            if (pool == 0) {
                for (int d1 = 0; d1 < N_level; ++d1) {
                    // levels_gibbs(save, d1) = level_values[d1];
                    levels_gibbs(save, d1) = ptr_level_values[d1];
                }
            } else {
                for (int d1 = 0; d1 < S; ++d1) {
                    for (int d2 = 0; d2 < C; ++d2) {
                        int idx = d1*C + d2;
                        probbase_gibbs(save, d1, d2) = ptr_probbase[idx];
                    }
                }
            }
        }
    }
    if (is_openva_app == false) {
        py::print("\nOverall acceptance ratio");
        for (int sub = 0; sub < N_sub; ++sub) {
            double ratio = std::ceil(100.0 * n_accept[sub] / (N_gibbs + 0.0)) / 100.0;
            py::print("Sub-population ", sub, ": ", ratio);
	}
	py::print("Organizing output, might take a moment...");
    }
    // The outcome is vector of
    //  0. scalar of N_thin
    //	1. CSMF at each iteration N_sub * C * N_thin
    //  2. Individual prob mean N * C
    //  4. last time configuration N_sub * (C + C + 1)
    //  3. probbase at each iteration
//    int N_out = 1 + N_sub * C * N_thin + N * C + N_sub * (C * 2 + 1);
//    if (pool == 0) {
//	N_out += N_level * N_thin;
//    } else {
//	N_out += S * C * N_thin;
//    }

    // save CSMF at each iteration
    // counter of which column is
    int counter = 0;
    // save N_thin;
    parameters(0) = N_thin;
    counter = 1;
    // save p_gibbs
    for (int sub = 0; sub < N_sub; ++sub) {
        for (int k = 0; k < N_thin; ++k) {
            for (int c = 0; c < C; ++c) {
                parameters(counter) = p_gibbs(k, sub, c);
                counter += 1;
            }
        }
    }
    // save pnb_gibbs, need normalize here
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            parameters(counter) = pnb_mean(n, c) / (N_thin + 0.0);
            counter += 1;
        }
    }
    // save probbase at each iteration
    if (pool != 0) {
        for (int c = 0; c < C; ++c) {
            for (int s = 0; s < S; ++s) {
                for (int k = 0; k < N_thin; ++k) {
                    parameters(counter) = probbase_gibbs(k, s, c);
                    counter += 1;
                }
            }
        }
    } else {
        for (int k = 0; k < N_thin; ++k) {
            // for (int l = 1; l <= N_level; ++l) {
            // parameters(counter) = levels_gibbs(k, l-1);
            for (int l = 0; l < N_level; ++l) {
                parameters(counter) = levels_gibbs(k, l);
                counter += 1;
            }
        }
    }

    // save output without N_thin rows
    // only using the first row
    // save mu_now
    for (int sub = 0; sub < N_sub; ++sub) {
        for (int c = 0; c < C; ++c) {
            parameters(counter) = mu_now(sub, c);
            counter += 1;
        }
    }
    // save sigma2_now
    for (int sub = 0; sub < N_sub; ++sub) {
        parameters(counter) = sigma2_now(sub);
        counter += 1;
    }
    // save theta_now
    for (int sub = 0; sub < N_sub; ++sub) {
        for (int c = 0; c < C; ++c) {
            parameters(counter) = theta_now(sub, c);
            counter += 1;
        }
    }
}


PYBIND11_MODULE(_sampler, m) {
    m.doc() = R"pbdoc(
        C++ Sampler for InSilicoVA 
        -----------------------

        .. currentmodule:: sampler

        .. autosummary::
           :toctree: _generate

           Sampler
    )pbdoc";

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<std::vector<int> &, py::array_t<int>,
             py::array_t<double>, py::array_t<int>,
             py::dict,
             py::array_t<double>,
             py::array_t<int> &, py::array_t<int> &,
             py::array_t<int> & >())
        .def("fit", &Sampler::fit);
}
