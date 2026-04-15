import pandas as pd, numpy as np
from pathlib import Path
import html

src = Path("events_processed_clean.csv")
base = pd.read_csv(src)

# Parse dates
base['starts_dt'] = pd.to_datetime(base['starts_dt'], errors='coerce', utc=True)
base['start_date'] = pd.to_datetime(base['start_date'], errors='coerce')
base['category'] = base['category'].fillna('Не указано')
base['season'] = base['season'].fillna('unknown')
base['is_weekend'] = pd.to_numeric(base['is_weekend'], errors='coerce').fillna(0).astype(int)

# Seed
rng = np.random.default_rng(42)

# Synthetic city
cities = np.array(['Москва','Санкт-Петербург','Казань','Екатеринбург','Новосибирск','Омск','Другие'])
city_probs = np.array([0.30,0.14,0.06,0.06,0.06,0.04,0.34])
city_syn = rng.choice(cities, size=len(base), p=city_probs)

# Holiday flag (RU fixed-date federal holidays, simplified)
ru_holidays = {(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
               (2,23),(3,8),(5,1),(5,9),(6,12),(11,4)}
month = pd.to_numeric(base['start_month'], errors='coerce').fillna(base['start_date'].dt.month).astype(int)
day = base['start_date'].dt.day.fillna(1).astype(int)
holiday_syn = np.array([1 if (int(m), int(d)) in ru_holidays else 0 for m, d in zip(month, day)], dtype=int)

# Segment by category
def segment_from_category(cat: str) -> str:
    c = (cat or "").lower()
    if 'конц' in c: return 'concert'
    if 'театр' in c: return 'theatre'
    if 'выстав' in c: return 'exhibition'
    if 'спорт' in c: return 'sport'
    if 'бизнес' in c: return 'business'
    if 'для детей' in c: return 'kids'
    if 'экскурс' in c or 'путеше' in c: return 'tour'
    if 'ит и интернет' in c or 'it' in c: return 'it'
    if 'психолог' in c: return 'psy'
    return 'other'
segment = base['category'].map(segment_from_category)

# Price synthesis: truncated normal around segment and city anchors
price_mu = {'concert':4907,'theatre':4151,'exhibition':1453,'sport':1732,'business':6000,'kids':2500,'tour':1800,'it':3500,'psy':2000,'other':2200}
price_sigma = {k: v*0.35 for k,v in price_mu.items()}
city_mult = {'Москва':1.20,'Санкт-Петербург':1.10,'Казань':0.95,'Екатеринбург':0.95,'Новосибирск':0.90,'Омск':0.85,'Другие':0.85}

mu = segment.map(price_mu).astype(float) * pd.Series(city_syn).map(city_mult).astype(float)
sig = segment.map(price_sigma).astype(float)

is_free_syn = rng.binomial(1, p=0.55, size=len(base)).astype(int)
price_syn = rng.normal(mu, sig)
price_syn = np.clip(price_syn, 0, None)
price_syn = np.where(is_free_syn==1, 0, price_syn)
price_syn = np.round(price_syn, 0).astype(int)

# Capacity synthesis: lognormal, clipped
seg_cap_scale = {'concert':1.25,'theatre':0.90,'exhibition':0.70,'sport':1.10,'business':0.95,'kids':0.65,'tour':0.60,'it':0.80,'psy':0.55,'other':0.70}
base_cap = rng.lognormal(mean=6.0, sigma=0.55, size=len(base))
capacity_syn = base_cap * segment.map(seg_cap_scale).astype(float)
capacity_syn = np.clip(capacity_syn, 50, 8500)
capacity_syn = np.round(capacity_syn).astype(int)

# Lead days
lead_days_syn = rng.normal(loc=18, scale=10, size=len(base))
lead_days_syn = np.clip(lead_days_syn, 1, 90)
lead_days_syn = np.round(lead_days_syn).astype(int)

# Ad budget: % of expected revenue + noise with floors
expected_revenue = price_syn.astype(float) * capacity_syn.astype(float) * 0.65
share = rng.uniform(0.12, 0.32, size=len(base))
budget = expected_revenue * share
min_floor = np.where((segment=='business') & (city_syn=='Москва'), 100_000,
            np.where((segment=='business') & (city_syn!='Москва'), 50_000, 5_000))
budget = np.maximum(budget, min_floor)
budget *= rng.lognormal(mean=0.0, sigma=0.35, size=len(base))
ad_budget_syn = np.round(budget, -1).astype(int)

# Participants synthesis via logistic demand and attendance rate
def sigmoid(x): return 1/(1+np.exp(-x))
cat_bias = segment.map({'concert':0.8,'theatre':0.6,'exhibition':0.2,'sport':0.5,'business':0.1,'kids':0.3,'tour':0.25,'it':0.15,'psy':0.05,'other':0.15}).astype(float)
city_bias = pd.Series(city_syn).map({'Москва':0.35,'Санкт-Петербург':0.25,'Казань':0.10,'Екатеринбург':0.10,'Новосибирск':0.08,'Омск':0.05,'Другие':0.00}).astype(float).to_numpy()
weekend = base['is_weekend'].astype(float).to_numpy() * 0.18
holiday = holiday_syn.astype(float) * 0.10
season_bias = base['season'].map({'winter':0.05,'spring':0.08,'summer':-0.12,'autumn':0.02}).fillna(0).astype(float).to_numpy()
log_budget = np.log1p(ad_budget_syn.astype(float))
log_price = np.log1p(price_syn.astype(float))

score = (-0.8
         + cat_bias.to_numpy()
         + city_bias
         + weekend
         + holiday
         + season_bias
         + 0.10*log_budget
         - 0.18*log_price
         + 0.015*lead_days_syn.astype(float))
reg_frac = sigmoid(score)
registrations_syn = capacity_syn.astype(float) * reg_frac

att_rate = np.where(is_free_syn==1, rng.normal(0.78, 0.07, size=len(base)), rng.normal(0.86, 0.05, size=len(base)))
att_rate = np.clip(att_rate, 0.55, 0.98)

participants_syn = registrations_syn * att_rate
noise = rng.normal(0, np.sqrt(np.maximum(participants_syn,1))*2.2, size=len(base))
participants_syn = np.clip(participants_syn + noise, 0, capacity_syn.astype(float))
participants_syn = np.round(participants_syn).astype(int)

final = base.copy()
final['city'] = city_syn
final['holiday'] = holiday_syn
final['price'] = price_syn
final['is_free'] = is_free_syn
final['ad_budget'] = ad_budget_syn
final['capacity'] = capacity_syn
final['lead_days'] = lead_days_syn
final['registrations'] = np.round(registrations_syn).astype(int)
final['participants'] = participants_syn

# Keep a clean ML-ready set
cols = [
    'event_id','name_clean','url','starts_at',
    'start_year','start_month','start_dow','start_hour',
    'season','is_weekend','holiday',
    'category','city',
    'price','is_free','ad_budget','capacity','lead_days',
    'registrations','participants'
]
final_ml = final[cols].copy()

# Sanity: no missing in required cols
assert final_ml.isna().sum().sum() == 0

out = Path("events_final_dataset_synthetic.csv")
final_ml.to_csv(out, index=False, encoding='utf-8')
out.exists(), out.stat().st_size, final_ml.head(3)

