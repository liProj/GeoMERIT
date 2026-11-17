"""Decision selection on inner out-of-fold predictions only."""
from itertools import product
import numpy as np
from .labels import TAIL_IDS


def confusion(y, pred):
    return np.bincount(12*np.asarray(y, int)+np.asarray(pred, int), minlength=144).reshape(12, 12)


def scores_from_cm(cm, cost):
    support, proposed = cm.sum(axis=1), cm.sum(axis=0)
    f1 = np.divide(2*np.diag(cm), support+proposed, out=np.zeros(12), where=(support+proposed)>0)
    n = cm.sum()
    return {'weighted_f1': float(f1@support/max(n, 1)), 'macro_f1': float(f1.mean()),
            'tail_f1': float(f1[TAIL_IDS].mean()), 'penalty': -float((cm*cost).sum()/max(n, 1)),
            'accuracy': float(np.trace(cm)/max(n, 1))}


def normalize(p):
    p = np.asarray(p, dtype=np.float32)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)


def prior(y, smoothing=1.):
    counts = np.bincount(np.asarray(y, int), minlength=12).astype(np.float32) + smoothing
    return counts/counts.sum()


def geometric_pool(components, alpha):
    names = [n for n in ['lightgbm', 'xgboost', 'catboost'] if n in components]
    if len(names) == 1: return normalize(components[names[0]])
    weights = np.asarray(alpha, float)[:len(names)]; weights /= weights.sum()
    logp = np.zeros_like(components[names[0]], dtype=np.float32)
    supported = np.zeros_like(logp, dtype=bool)
    for name, weight in zip(names, weights):
        p = components[name]
        logp += weight * np.log(np.maximum(p, 1e-30))
        supported |= p > 0
    p = np.exp(logp-logp.max(axis=1, keepdims=True))
    p[~supported] = 0
    return normalize(p)


def posterior(components, selection):
    flat = geometric_pool(components, selection['alpha'])
    mu = selection['mu']
    return normalize((1-mu)*flat+mu*components['hierarchy']) if 'hierarchy' in components else flat


def adjust(p, priors, experts, setting):
    # This is prior adjustment, not temperature calibration. Zero support remains zero.
    q = p * np.power(np.maximum(priors, 1e-30), -setting['tau'])
    ordinary = [c for c in TAIL_IDS if c != 9]
    for cols, gamma, theta in [(ordinary, setting['tail_gamma'], setting['tail_threshold']),
                               ([9], setting['dolomite_gamma'], setting['dolomite_threshold'])]:
        if gamma:
            q[:, cols] *= 1 + gamma*experts[:, cols]*(experts[:, cols] >= theta)
    return normalize(q)


def decide(p, cost, rule='bayes'):
    if rule == 'argmax': return np.argmax(p, axis=1).astype(np.int16)
    if rule != 'bayes': raise ValueError(rule)
    return np.argmin(p @ cost, axis=1).astype(np.int16)


def choose_blend(components, y_inner, config, cost):
    if 'hierarchy' not in components:
        return {'alpha': [1., 1., 1.], 'mu': 0., 'source': 'single fixed backbone'}
    best, key_best = None, None
    for alpha in config['selection']['ensemble_alpha']:
        flat = geometric_pool(components, alpha)
        for mu in config['selection']['hierarchy_mu']:
            p = normalize((1-mu)*flat+mu*components['hierarchy'])
            s = scores_from_cm(confusion(y_inner, p.argmax(axis=1)), cost)
            key = (s['macro_f1'], s['weighted_f1'], s['penalty'])
            if key_best is None or key > key_best:
                best, key_best = {'alpha': alpha, 'mu': float(mu), 'source': 'inner OOF only', 'inner_score': s}, key
    return best


def gate_grid(gammas, thresholds):
    return [(g, t) for g in gammas for t in (thresholds[:1] if g == 0 else thresholds)]


def choose_decision(p, experts, priors, y_inner, config, cost, allow_experts=True):
    cfg = config['selection']
    ordinary = gate_grid(cfg['tail_gamma'], cfg['tail_threshold']) if allow_experts else [(0., .5)]
    dolomite = gate_grid(cfg['dolomite_gamma'], cfg['dolomite_threshold']) if allow_experts else [(0., .5)]
    best, key_best, history = None, None, []
    for tau, (gamma, theta), (dg, dt) in product(cfg['tau'], ordinary, dolomite):
        setting = dict(tau=float(tau), tail_gamma=float(gamma), tail_threshold=float(theta),
                       dolomite_gamma=float(dg), dolomite_threshold=float(dt))
        pred = decide(adjust(p, priors, experts, setting), cost)
        score = scores_from_cm(confusion(y_inner, pred), cost)
        key = (score['weighted_f1'], score['penalty'], score['macro_f1'])
        history.append(dict(parameters=setting, scores=score))
        if key_best is None or key > key_best:
            best, key_best = setting, key
    return best, history


def evaluated_predictions(components, selected, priors, cost):
    p = posterior(components, selected['blend'])
    decision = selected['decision']
    experts = components['experts']
    full = adjust(p, priors, experts, decision)
    output = {'full': decide(full, cost), 'posterior_argmax': decide(p, cost, 'argmax'),
              'bayes_only': decide(p, cost), 'adjusted_argmax': decide(full, cost, 'argmax')}
    no_expert = dict(decision, tail_gamma=0., dolomite_gamma=0.)
    output['no_experts_conditional'] = decide(adjust(p, priors, experts, no_expert), cost)
    output['no_prior_conditional'] = decide(adjust(p, priors, experts, dict(decision, tau=0.)), cost)
    if 'hierarchy' in components:
        # Conditional interventions hold the selected decoder fixed and are labelled accordingly.
        flat = geometric_pool(components, selected['blend']['alpha'])
        for name, prob in [('flat_conditional', flat), ('soft_hierarchy_conditional', components['hierarchy']),
                           ('hard_hierarchy_conditional', components['hard_hierarchy'])]:
            output[name] = decide(adjust(prob, priors, experts, decision), cost)
        for name in ['lightgbm', 'xgboost', 'catboost']:
            if name in components: output[name+'_argmax'] = decide(components[name], cost, 'argmax')
    return output, full
