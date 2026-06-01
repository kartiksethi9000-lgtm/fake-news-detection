import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import string
import warnings
warnings.filterwarnings('ignore')



real_news = [
    # POLITICS
    "Parliament passes new budget bill after lengthy debate among lawmakers",
    "Government announces infrastructure spending plan worth billions for roads",
    "Prime minister meets foreign leaders at international summit on trade",
    "Election commission announces dates for upcoming general elections",
    "Supreme court rules on constitutional rights case after months of hearings",
    "Senate approves new education reform bill with bipartisan support",
    "President signs executive order on immigration policy reform",
    "Government releases annual report on economic performance indicators",
    "Opposition party calls for inquiry into government spending practices",
    "New legislation passed to strengthen data privacy rights for citizens",
    "Congress approves emergency funding for disaster relief operations",
    "Federal reserve raises interest rates to combat rising inflation",
    "Trade agreement signed between nations to reduce import tariffs",
    "Government announces new policy for renewable energy investment",
    "Local elections held across major cities with record voter turnout",

    # HEALTH & MEDICINE
    "Scientists develop new vaccine showing 94 percent effectiveness in trials",
    "WHO announces updated guidelines for antibiotic use to prevent resistance",
    "New cancer treatment shows promising results in clinical trial phase three",
    "Researchers find link between sleep deprivation and increased disease risk",
    "Hospital introduces robotic surgery system improving patient outcomes",
    "Study confirms regular exercise reduces risk of heart disease significantly",
    "New diabetes medication approved by drug regulatory authority after review",
    "Mental health awareness campaign launched across schools nationwide",
    "Doctors warn about rising cases of antibiotic resistant infections globally",
    "Research shows Mediterranean diet reduces risk of cardiovascular disease",
    "New blood test can detect Alzheimer's disease ten years before symptoms",
    "Scientists discover mechanism behind autoimmune disease progression",
    "Flu vaccine effectiveness this season estimated at sixty percent by CDC",
    "Study links air pollution to increased risk of dementia in older adults",
    "New surgical technique reduces recovery time for knee replacement patients",

    # SCIENCE & TECHNOLOGY
    "NASA confirms water ice found in permanently shadowed craters on moon",
    "Scientists detect gravitational waves from merging neutron star collision",
    "New study reveals ancient human migration patterns through DNA analysis",
    "Researchers develop biodegradable plastic alternative from plant material",
    "Tech company releases open source artificial intelligence language model",
    "Scientists discover new species of deep sea fish near hydrothermal vents",
    "Space telescope captures detailed images of galaxy formation in early universe",
    "Breakthrough in quantum computing achieves new processing speed record",
    "Researchers develop solar panel with forty percent energy conversion rate",
    "New battery technology could triple electric vehicle range on single charge",
    "Scientists sequence genome of newly discovered ancient human ancestor",
    "Study reveals microplastics found in human bloodstream for first time",
    "Astronomers confirm existence of exoplanet in habitable zone of nearby star",
    "New algorithm detects early signs of diabetic retinopathy from eye scans",
    "Fusion energy experiment achieves net energy gain milestone for first time",

    # BUSINESS & ECONOMY
    "Stock market rises after positive employment data released by government",
    "Major tech company reports record quarterly earnings beating analyst estimates",
    "Central bank holds interest rates steady amid economic uncertainty globally",
    "Retail sales increase by three percent in strongest month of the year",
    "Oil prices fall as OPEC members disagree on production quota levels",
    "Startup raises fifty million dollars in series B funding round from investors",
    "Manufacturing output rises for third consecutive month according to survey",
    "Consumer confidence index reaches highest level in five years this month",
    "Major airline announces new routes expanding international travel options",
    "Housing market shows signs of cooling as mortgage rates continue rising",
    "Company announces merger creating one of largest firms in the industry",
    "Unemployment rate falls to historic low as job creation continues strong",
    "Cryptocurrency exchange files for bankruptcy after significant losses",
    "Supply chain disruptions ease as shipping costs return toward normal levels",
    "Small business lending increases as banks loosen credit requirements",

    # SPORTS
    "National team wins championship after defeating rivals in final match",
    "Athlete breaks world record in hundred meter sprint at international meet",
    "Club signs star player for record transfer fee ahead of new season",
    "Olympic committee announces host city for upcoming summer games",
    "Tennis champion wins fourth grand slam title of career at major tournament",
    "Football league announces expansion with two new franchises added",
    "Coach fired after team records worst losing streak in franchise history",
    "Marathon runner completes race in new personal best time despite injury",
    "Stadium renovation project approved creating thousands of construction jobs",
    "Young player becomes youngest scorer in league history at age seventeen",
    "Sports governing body bans athlete for doping violation after positive test",
    "Team advances to semifinal after dramatic penalty shootout victory",
    "New sports science research helps athletes recover faster from injuries",
    "League introduces new technology to assist referees in making decisions",
    "Cricket world cup final attracts record television audience worldwide",

    # ENVIRONMENT
    "Global temperatures rise by one point two degrees above pre-industrial levels",
    "New report shows Arctic sea ice at second lowest extent ever recorded",
    "Countries agree to phase out coal power by twenty thirty five at summit",
    "Endangered species population increases following conservation program",
    "Scientists document coral reef recovery after decade of restoration effort",
    "Wildfire season causes record destruction across western regions this year",
    "New study measures dramatic decline in insect populations over decades",
    "City implements congestion charge reducing air pollution by thirty percent",
    "Ocean cleanup project removes thousands of tons of plastic from Pacific",
    "Renewable energy surpasses coal in electricity generation for first time",
    "Amazon deforestation rate falls after government strengthens enforcement",
    "New satellite data reveals extent of glacial retreat over past century",
    "Electric vehicle sales double as governments offer purchase incentives",
    "Scientists warn of accelerating biodiversity loss threatening ecosystems",
    "Flood management infrastructure investment reduces disaster impact in region",

    # WORLD NEWS
    "United nations peacekeeping mission deployed to conflict affected region",
    "International aid organizations deliver food supplies to drought hit areas",
    "Diplomatic talks resume between nations after months of stalled negotiations",
    "Earthquake measuring six point five strikes coastal region injuring dozens",
    "Refugee resettlement program approved allowing thousands to seek asylum",
    "Foreign minister visits allied nation to strengthen bilateral trade ties",
    "International court rules on territorial dispute between neighboring states",
    "Humanitarian corridor opened allowing civilians to leave conflict zone",
    "Global health organization warns of disease outbreak in affected region",
    "Developing nations receive debt relief package from international lenders",
    "Border talks produce agreement on shared water resource management plan",
    "International sanctions imposed on nation over human rights violations",
    "Peacekeeping force reports reduction in violence in monitored region",
    "Cultural exchange program launched between schools in partner countries",
    "Natural disaster response team deployed within hours of earthquake striking",

    # ENTERTAINMENT
    "Film wins best picture award at prestigious international ceremony event",
    "Streaming service announces cancellation of popular drama series",
    "Music artist releases debut album reaching top of charts in thirty countries",
    "Theater production receives critical acclaim after opening night performance",
    "Documentary about climate change wins award at film festival",
    "Video game studio announces sequel to best selling role playing game",
    "Famous author releases highly anticipated novel after decade long wait",
    "Concert tour breaks attendance records across all scheduled venues",
    "Television network announces adaptation of classic book series",
    "Podcast about true crime becomes most downloaded show of the year",

    # EDUCATION
    "University research team publishes findings on effective teaching methods",
    "Government announces increased funding for public school infrastructure",
    "Students perform better when schools start later according to new study",
    "Online learning platform sees surge in enrollment after partnership",
    "New literacy program helps struggling readers improve by two grade levels",
    "Teachers union reaches agreement on salary increases with school board",
    "Study finds smaller class sizes improve student academic performance",
    "University introduces new scholarship program for underprivileged students",
    "Research confirms preschool education has lifelong positive impact on children",
    "Schools adopt new curriculum focused on digital literacy and coding skills",
]

fake_news = [
    # POLITICS FAKE
    "Government secretly implanting tracking devices in new national ID cards",
    "President signs secret deal to surrender national sovereignty to world body",
    "Politicians drink the blood of children at secret underground ceremonies",
    "Election results completely fabricated by machines controlled by elites",
    "Government adding chemicals to drinking water to make population compliant",
    "Secret memo reveals plan to imprison all political opposition by next year",
    "Politician caught on tape admitting elections are staged performances",
    "Shadow government runs all world nations from hidden bunker underground",
    "Voting machines pre-programmed to always elect predetermined candidates",
    "Congress secretly voted to eliminate the constitution in closed session",
    "Government spraying population with mind control chemicals from aircraft",
    "All major political parties secretly controlled by same group of billionaires",
    "New law secretly passed that allows government to seize all private property",
    "Senator caught selling state secrets to foreign enemy in airport meeting",
    "Government preparing internment camps for citizens who refuse vaccines",

    # HEALTH FAKE
    "Drinking bleach mixed with lemon juice cures all known cancers permanently",
    "Doctors secretly paid to hide natural cure that eliminates diabetes forever",
    "Vaccines contain microchips that track your location and thoughts constantly",
    "Big pharma suppressing hundred year old cure for all diseases from public",
    "Eating magnets reverses aging process doctors do not want you knowing this",
    "Hospital patients being harvested for organs without their consent secretly",
    "New study proves eating clay cures autoimmune diseases doctors hide truth",
    "Drinking hydrogen peroxide daily prevents all viruses and bacteria forever",
    "Cancer is actually a fungus that can be cured with baking soda instantly",
    "Fluoride in water is poison designed to lower population intelligence levels",
    "Miracle herb from Amazon jungle cures HIV doctors suppressing discovery",
    "Sunscreen causes cancer and is conspiracy by dermatology industry exposed",
    "Mobile phones cause brain tumors doctors hiding truth for industry profits",
    "Raw garlic eaten every hour cures heart disease completely within days",
    "Autism caused by vaccines secret documents from pharmaceutical company leaked",

    # SCIENCE FAKE
    "Moon landing completely staged in Hollywood studio director admits finally",
    "NASA hiding giant alien structures found on surface of Mars from public",
    "Flat earth confirmed by leaked classified documents from space agencies",
    "Evolution is completely fabricated theory with no scientific evidence at all",
    "Scientists paid billions to fake climate change data by world governments",
    "Earth is actually hollow and civilizations live inside according to insider",
    "Dinosaurs never existed and fossils were planted by scientific establishment",
    "Time travel already invented and government using it to manipulate history",
    "Giants once roamed earth and Smithsonian institution destroying all evidence",
    "Scientists admit privately that gravity does not exist it is just pressure",
    "Ancient pyramids built by aliens not humans secret documents now revealed",
    "Parallel universe portal opened by scientists causing disasters worldwide",
    "Antarctica hiding advanced alien civilization under ice governments know",
    "Sun is actually a cold body that creates heat through electromagnetic field",
    "DNA evidence proves humans created by alien species two hundred years ago",

    # BUSINESS FAKE
    "Secret elite group controls all stock markets and profits from crashes",
    "Banks create money from nothing and secretly drain accounts while sleeping",
    "Federal reserve owned by foreign nation secretly draining American wealth",
    "Major corporation putting addictive substances in products to hook consumers",
    "Billionaires meeting secretly to plan economic collapse and buy assets cheap",
    "All major banks will collapse next week insider source reveals exclusively",
    "Gold standard secretly abandoned and all currency is now worthless paper",
    "Tech giants reading all private messages and selling to government agencies",
    "Major retailer caught putting tracking devices in clothing sold to customers",
    "Secret algorithm manipulates all cryptocurrency prices billionaires profit",

    # SPORTS FAKE
    "Professional sports leagues scripted and outcomes predetermined by owners",
    "Famous athlete secretly replaced by clone after real one died last year",
    "World cup matches fixed by international gambling syndicate insider reveals",
    "Olympic athletes given experimental drugs by governments without consent",
    "Major sports star faked injury to avoid being exposed as performance cheater",
    "Referee admits all championship games decided in advance by league officials",
    "Professional wrestler secretly killed during match covered up by promoters",
    "International sports body controlled by secret society that fixes all results",
    "Famous sports team uses supernatural rituals before games insider claims",
    "Drug testing in sports completely fake as labs are bribed to hide results",

    # ENVIRONMENT FAKE
    "Climate change completely fabricated by scientists to receive grant funding",
    "Chemtrails from aircraft spreading poison designed to reduce world population",
    "Electric cars actually worse for environment than petrol secret study reveals",
    "Volcanoes produce more carbon than all human activity combined experts hide",
    "Global warming is actually cooling and elites hiding data from population",
    "Wind turbines causing cancer in nearby residents government hiding evidence",
    "Water fluoridation secret program to make population sterile and controllable",
    "Trees are actually antennas planted by government to spy on citizens secretly",
    "Wildfires all started by government using directed energy weapons from space",
    "Oceans not rising satellite data manipulated by climate scientists for grants",

    # WORLD NEWS FAKE
    "Secret world government meeting in underground base planning population cull",
    "Country secretly invaded by hidden army that disguised as tourists entering",
    "World war three already started but media banned from reporting truth now",
    "Foreign nation has weather control weapon causing disasters in enemy states",
    "United nations planning to take over all sovereign nations by next decade",
    "Major country secretly bankrupt and hiding economic collapse from citizens",
    "Secret tunnel network connects all world capitals used by global elite only",
    "Alien beings have contacted world leaders and are advising their decisions",
    "Major terrorist attack staged by government to justify new surveillance laws",
    "Country discovered free energy device but global oil cartel had inventor killed",

    # ENTERTAINMENT FAKE
    "Famous actor is secretly a lizard person who shape shifts between appearances",
    "Hollywood stars participate in secret rituals to gain fame and success",
    "Popular singer died years ago and replaced by look alike clone performing",
    "Major film studio secretly embedding subliminal messages in all their movies",
    "Celebrity faked own death and living under new identity in foreign country",
    "Rock music secretly designed to corrupt youth by government psychological ops",
    "Famous musician sold soul to devil in exchange for talent and chart success",
    "All award shows rigged and decided by secret committee not actual voting",
    "Child actors systematically abused by studio executives throughout industry",
    "Popular television show contains hidden satanic symbols only experts notice",

    # CONSPIRACY GENERAL
    "Reptilian alien beings control all major world governments from the shadows",
    "Secret society over three hundred years old controls all world economies",
    "Population control chips to be mandatory in all citizens within five years",
    "New world order finalizing plan to reduce world population by ninety percent",
    "Birds are government surveillance drones and not real living creatures",
    "Simulation theory confirmed we all live inside computer program now proven",
    "Secret base on moon used by military forces hidden from public since sixties",
    "Ancient advanced civilization destroyed when elites activated global weapon",
    "All world religions created by same elite group to control human population",
    "Reality television shows all scripted with paid actors not real people used",
]

# ──────────────────────────────────────────────────────────────
texts  = real_news + fake_news
labels = [1] * len(real_news) + [0] * len(fake_news)

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 60)
print("   FAKE NEWS DETECTION SYSTEM — Enhanced Version")
print("=" * 60)
print(f"\n📊 Dataset Overview")
print(f"   Total samples : {len(df)}")
print(f"   Real news     : {sum(df.label == 1)}")
print(f"   Fake news     : {sum(df.label == 0)}")
print(f"   Categories    : Politics, Health, Science, Tech,")
print(f"                   Sports, Business, World, Environment")

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\n📂 Train/Test Split (80/20)")
print(f"   Training : {len(X_train)} samples")
print(f"   Testing  : {len(X_test)} samples")

# ── TF-IDF ────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=1,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n📐 TF-IDF Feature Extraction")
print(f"   Vocabulary size : {len(tfidf.vocabulary_)}")
print(f"   Feature matrix  : {X_train_tfidf.shape}")

# ── Models ────────────────────────────────────────────────────
lr  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
nb  = MultinomialNB(alpha=0.1)

models = {"Logistic Regression": lr, "Naive Bayes": nb}
results = {}

print(f"\n🤖 Model Training & Evaluation")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy}
    report = classification_report(y_test, y_pred,
                                   target_names=['Fake','Real'],
                                   output_dict=True)
    print(f"\n  ▶ {name}  —  Accuracy: {accuracy*100:.2f}%")
    print(f"    Fake — P:{report['Fake']['precision']:.2f}  R:{report['Fake']['recall']:.2f}  F1:{report['Fake']['f1-score']:.2f}")
    print(f"    Real — P:{report['Real']['precision']:.2f}  R:{report['Real']['recall']:.2f}  F1:{report['Real']['f1-score']:.2f}")

best_name  = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']

print(f"\n🏆 Best Model : {best_name}")
print(f"   Accuracy   : {results[best_name]['accuracy']*100:.2f}%")

# ── Predict Function ──────────────────────────────────────────
def predict_news(article):
    cleaned    = preprocess_text(article)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    probability = best_model.predict_proba(vectorized)[0]
    label      = "✅ REAL NEWS" if prediction == 1 else "❌ FAKE NEWS"
    confidence = max(probability) * 100
    return label, confidence

# ── Live Tests ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  🔍 LIVE PREDICTIONS — All Category Tests")
print(f"{'=' * 60}")

test_articles = [
    ("Politics",     "Government passes new law improving healthcare access for citizens"),
    ("Politics",     "Politicians secretly drink blood of children in underground rituals"),
    ("Health",       "New study confirms exercise reduces risk of heart disease"),
    ("Health",       "Drinking bleach cures cancer doctors hiding this truth"),
    ("Science",      "Scientists discover water ice in craters on surface of moon"),
    ("Science",      "Moon landing was faked in Hollywood studio director admits"),
    ("Business",     "Stock market rises after positive jobs data released today"),
    ("Business",     "Secret elite group controls all stock markets and crashes them"),
    ("Sports",       "Athlete breaks world record at international championship"),
    ("Sports",       "All professional sports matches scripted and predetermined"),
    ("Environment",  "Renewable energy surpasses coal in electricity generation"),
    ("Environment",  "Climate change completely fabricated by scientists for funding"),
    ("World",        "United nations peacekeeping mission deployed to conflict zone"),
    ("World",        "Reptilian aliens control all major world governments secretly"),
    ("Political",    "President has influence on prices through tariffs and policy"),
    ("CBS Claim",    "The president of the United States has nothing to do with the price of bacon or eggs or gas"),
]

for category, article in test_articles:
    label, confidence = predict_news(article)
    print(f"\n  [{category}]")
    print(f"  Article : {article[:65]}...")
    print(f"  Result  : {label}  (Confidence: {confidence:.1f}%)")

print(f"\n{'=' * 60}\n")

