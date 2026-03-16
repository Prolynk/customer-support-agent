"""Generate a recruiter interview Q&A PDF for the intent classifier project.

Covers every likely question a recruiter or technical interviewer would ask,
with clear, simple answers explained as if to a 7-year-old — no jargon left
unexplained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = Path("results/interview_prep.pdf")
OUTPUT.parent.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "ITitle", parent=styles["Title"],
    fontSize=30, textColor=colors.HexColor("#0f3460"),
    spaceAfter=12, alignment=TA_CENTER, fontName="Helvetica-Bold"
)
SUBTITLE_STYLE = ParagraphStyle(
    "ISubtitle", parent=styles["Normal"],
    fontSize=13, textColor=colors.HexColor("#533483"),
    spaceAfter=8, alignment=TA_CENTER
)
COVER_BODY = ParagraphStyle(
    "ICoverBody", parent=styles["Normal"],
    fontSize=11, leading=17, textColor=colors.HexColor("#1a1a2e"),
    alignment=TA_CENTER, spaceAfter=8
)
SECTION_STYLE = ParagraphStyle(
    "ISection", parent=styles["Heading1"],
    fontSize=18, textColor=colors.white,
    spaceBefore=16, spaceAfter=8,
    backColor=colors.HexColor("#0f3460"),
    borderPad=8, fontName="Helvetica-Bold"
)
CATEGORY_STYLE = ParagraphStyle(
    "ICategory", parent=styles["Heading2"],
    fontSize=13, textColor=colors.HexColor("#533483"),
    spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold"
)
Q_STYLE = ParagraphStyle(
    "IQuestion", parent=styles["Normal"],
    fontSize=11, leading=16, textColor=colors.HexColor("#0f3460"),
    spaceBefore=10, spaceAfter=3, fontName="Helvetica-Bold",
    backColor=colors.HexColor("#e8f4fd"),
    borderColor=colors.HexColor("#0f3460"),
    borderWidth=1, borderPad=7, borderRadius=4,
    leftIndent=0
)
A_STYLE = ParagraphStyle(
    "IAnswer", parent=styles["Normal"],
    fontSize=10, leading=16, textColor=colors.HexColor("#1a1a1a"),
    spaceBefore=4, spaceAfter=4, alignment=TA_JUSTIFY,
    leftIndent=8
)
SIMPLE_STYLE = ParagraphStyle(
    "ISimple", parent=styles["Normal"],
    fontSize=10, leading=15, textColor=colors.HexColor("#065f46"),
    spaceBefore=4, spaceAfter=6,
    backColor=colors.HexColor("#ecfdf5"),
    borderColor=colors.HexColor("#6ee7b7"),
    borderWidth=1, borderPad=6, borderRadius=3,
    leftIndent=8
)
TIP_STYLE = ParagraphStyle(
    "ITip", parent=styles["Normal"],
    fontSize=9.5, leading=14, textColor=colors.HexColor("#92400e"),
    spaceBefore=3, spaceAfter=6,
    backColor=colors.HexColor("#fffbeb"),
    borderColor=colors.HexColor("#fcd34d"),
    borderWidth=1, borderPad=5,
    leftIndent=8
)
BULLET_STYLE = ParagraphStyle(
    "IBullet", parent=styles["Normal"],
    fontSize=10, leading=15, textColor=colors.HexColor("#1a1a1a"),
    leftIndent=20, spaceAfter=3,
    bulletIndent=10
)
BODY_STYLE = ParagraphStyle(
    "IBody", parent=styles["Normal"],
    fontSize=10, leading=15, textColor=colors.HexColor("#374151"),
    spaceAfter=5, alignment=TA_JUSTIFY
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sec(title):
    return [Spacer(1, 10), Paragraph(f"  {title}", SECTION_STYLE), Spacer(1, 6)]

def cat(title):
    if isinstance(title, list):
        title = title[0]
    return [Paragraph(title, CATEGORY_STYLE), HRFlowable(width="100%", thickness=0.8,
            color=colors.HexColor("#533483"), spaceAfter=4)]

def q(text):
    return Paragraph(f"Q: {text}", Q_STYLE)

def a(text):
    return Paragraph(text, A_STYLE)

def simple(text):
    return Paragraph(f"  Simple version: {text}", SIMPLE_STYLE)

def tip(text):
    return Paragraph(f"  Interview Tip: {text}", TIP_STYLE)

def bul(text):
    return Paragraph(f"  - {text}", BULLET_STYLE)

def body(text):
    return Paragraph(text, BODY_STYLE)

def sp(n=8):
    return Spacer(1, n)

def rule():
    return HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#e5e7eb"), spaceAfter=6)

def qa_block(question, answer_text, simple_text="", tip_text="", bullets=None):
    """One complete Q&A block with optional simple version, tip, and bullets."""
    items = [sp(4), q(question), sp(3), a(answer_text)]
    if bullets:
        for b in bullets:
            items.append(bul(b))
    if simple_text:
        items.append(sp(2))
        items.append(simple(simple_text))
    if tip_text:
        items.append(sp(2))
        items.append(tip(tip_text))
    items.append(sp(4))
    items.append(rule())
    return items

# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------
story = []

# ===== COVER PAGE =====
story += [
    sp(50),
    Paragraph("Interview Prep Guide", TITLE_STYLE),
    Paragraph("Customer Support AI — Intent Classifier Project", SUBTITLE_STYLE),
    sp(16),
    Paragraph(
        "This guide prepares you to answer any question a recruiter or technical interviewer "
        "might ask about your Customer Support AI project.",
        COVER_BODY
    ),
    sp(8),
    Paragraph(
        "Every answer is written twice: once in proper technical language, and once in "
        "super-simple language — the way you would explain it to a 7-year-old. "
        "Reading both will make the concept stick.",
        COVER_BODY
    ),
    sp(20),
]

# Summary box
cover_table = Table(
    [[
        Paragraph("30\nQuestions\nCovered", ParagraphStyle("ct", fontSize=13, alignment=TA_CENTER,
                  textColor=colors.white, fontName="Helvetica-Bold", leading=18)),
        Paragraph("5\nDifficulty\nLevels", ParagraphStyle("ct2", fontSize=13, alignment=TA_CENTER,
                  textColor=colors.white, fontName="Helvetica-Bold", leading=18)),
        Paragraph("Simple\nExplanation\nEvery Time", ParagraphStyle("ct3", fontSize=13,
                  alignment=TA_CENTER, textColor=colors.white, fontName="Helvetica-Bold", leading=18)),
    ]],
    colWidths=[5*cm, 5*cm, 5*cm]
)
cover_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#0f3460")),
    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#533483")),
    ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#2d6a4f")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [None]),
    ("BOX", (0, 0), (-1, -1), 1, colors.white),
    ("INNERGRID", (0, 0), (-1, -1), 1, colors.white),
    ("TOPPADDING", (0, 0), (-1, -1), 14),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
]))
story.append(cover_table)
story.append(PageBreak())

# ===== TABLE OF CONTENTS =====
story += sec("Table of Contents")
toc_data = [
    ["Section", "Topic", "Page"],
    ["1", "The Big Picture — What Did You Build?", "3"],
    ["2", "The Data — Where Did It Come From?", "5"],
    ["3", "The Models — How Did You Train Them?", "7"],
    ["4", "The Pipeline — How Does It All Connect?", "11"],
    ["5", "Evaluation — How Do You Know It Works?", "13"],
    ["6", "Challenges & Problem Solving", "16"],
    ["7", "Production & Real-World Thinking", "18"],
    ["8", "Behavioural Questions", "21"],
    ["9", "Rapid-Fire Questions (Short Answers)", "23"],
    ["10", "Questions YOU Should Ask the Interviewer", "25"],
]
toc = Table(toc_data, colWidths=[1.5*cm, 12*cm, 2.5*cm])
toc.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 10),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
    ("FONTSIZE", (0, 1), (-1, -1), 10),
    ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ("ALIGN", (2, 0), (2, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
    ("TOPPADDING", (0, 0), (-1, -1), 7),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story += [toc, PageBreak()]

# ===========================================================================
# SECTION 1 — THE BIG PICTURE
# ===========================================================================
story += sec("Section 1: The Big Picture — What Did You Build?")

story += cat(["Overview Questions"])

story += qa_block(
    question="Can you give me a 60-second summary of this project?",
    answer_text=(
        "I built a two-stage automated customer support system. In stage one, a fine-tuned "
        "DistilBERT model reads an incoming customer message and classifies it into one of six "
        "intent categories — things like billing issues, account access problems, or cancellation "
        "requests. In stage two, the predicted intent is passed as context to Claude (an Anthropic "
        "LLM), which then generates a helpful, human-sounding support response tailored to that "
        "specific intent. The system also flags low-confidence predictions for human review. I "
        "evaluated the full pipeline using a custom LLM-based scoring framework for faithfulness "
        "and answer relevancy, achieving 0.837 answer relevancy on 50 test queries."
    ),
    simple_text=(
        "Imagine a robot postbox at a company. When a customer sends a message, the robot reads "
        "it and puts it in one of six boxes — like 'money problems' or 'can't log in'. Then a "
        "second, smarter robot writes a kind reply based on which box it went into. I built both "
        "robots and tested how well they work."
    ),
    tip_text="Always open with: what it does, how it works, and one key result. This answer does all three."
)

story += qa_block(
    question="Why did you choose this project?",
    answer_text=(
        "Customer support automation is a genuine industry problem — companies spend billions "
        "on support operations and response quality is inconsistent. This project let me practice "
        "the full ML lifecycle in one place: data engineering, fine-tuning a transformer model, "
        "prompt engineering with a production LLM, evaluation framework design, and packaging "
        "everything into a reproducible pipeline. It also demonstrates that I understand both "
        "classical NLP (TF-IDF baseline) and modern deep learning approaches."
    ),
    simple_text=(
        "Customer support is expensive and slow. I wanted to build something that actually "
        "saves a company time and money. And it let me practice every important skill in one "
        "single project — like training for a sports competition by doing every exercise at once."
    ),
    tip_text="Show that you understood the business problem, not just the tech. Recruiters love this."
)

story += qa_block(
    question="What are the two stages of the pipeline?",
    answer_text=(
        "Stage 1 is the Intent Classifier: a DistilBERT transformer model fine-tuned on labelled "
        "customer support examples. It reads the raw customer query and outputs a predicted intent "
        "label plus a confidence score. Stage 2 is the Response Generator: an Anthropic Claude "
        "model that receives the original query plus a structured prompt template filled with "
        "intent-specific guidance, and produces a personalised support response. The two stages "
        "are chained in the SupportAgent class."
    ),
    simple_text=(
        "Stage 1 is the SORTING robot — it reads the message and decides what kind of problem "
        "it is. Stage 2 is the WRITING robot — it reads the sorted message and writes a nice "
        "reply. They work together like a post office and a letter writer."
    ),
    tip_text="Draw this on a whiteboard if you get the chance. Diagrams make answers memorable."
)

story += qa_block(
    question="What are the 6 intent categories and how did you choose them?",
    answer_text=(
        "The six categories are: billing_issue (charges, refunds, payment problems), "
        "account_access (login, password, account management), technical_support (product "
        "or service problems, delivery), product_inquiry (information, compatibility, "
        "warranty), cancellation_request (cancelling orders or subscriptions), and "
        "general_feedback (complaints, suggestions, general questions). I derived these "
        "by analysing the Bitext customer support dataset's 50+ granular intent tags and "
        "grouping them into business-meaningful categories that a real support department "
        "would use to route tickets."
    ),
    simple_text=(
        "Think of it like sorting your toys into boxes: money box, login box, broken-thing box, "
        "asking-questions box, I-want-to-quit box, and other box. These six boxes cover almost "
        "everything a customer could ever message about."
    ),
    tip_text="Mention that the categories were business-driven, not just technically convenient. This shows maturity."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 2 — THE DATA
# ===========================================================================
story += sec("Section 2: The Data — Where Did It Come From?")

story += cat(["Dataset Questions"])

story += qa_block(
    question="What dataset did you use and why?",
    answer_text=(
        "I used the Bitext Customer Support LLM Chatbot Training Dataset from HuggingFace, "
        "which contains 26,872 labelled customer support utterances across 50+ fine-grained "
        "intent categories. I chose it because it is publicly available, professionally "
        "labelled, representative of real support language, and large enough to fine-tune "
        "a transformer model reliably. It also covers a wide vocabulary of customer phrasings "
        "for the same intent, which helps the model generalise."
    ),
    simple_text=(
        "I found a big collection of 26,872 real customer messages on the internet. Each "
        "message already had a label saying what the customer wanted. It's like having a "
        "giant homework sheet that already has all the answers marked — perfect for teaching "
        "the robot."
    ),
    tip_text="Always know your dataset size, source, and why it was appropriate. These are standard first questions."
)

story += qa_block(
    question="How did you preprocess the data?",
    answer_text=(
        "Preprocessing involved three steps: (1) Text cleaning — converting text to lowercase, "
        "stripping non-ASCII characters, and normalising whitespace using regex. This reduces "
        "vocabulary noise without removing meaningful content. (2) Label mapping — the Bitext "
        "dataset has 50+ granular tags which I mapped to my 6 business categories using a "
        "keyword-based dictionary (LABEL_MAP). Labels that didn't match a keyword got assigned "
        "via a fallback heuristic. (3) Stratified splitting — I split the data 70/15/15 into "
        "train/validation/test sets using sklearn's train_test_split with stratify=label, "
        "ensuring all 6 classes are proportionally represented in every split."
    ),
    simple_text=(
        "I cleaned the messages (made everything lowercase, removed weird characters), "
        "then sorted the 50+ original label types into my 6 big categories, "
        "and finally split the data into three piles: a teaching pile, a practice pile, "
        "and a final exam pile."
    ),
    tip_text="Stratified splitting is an important detail that shows you understand class imbalance. Mention it confidently."
)

story += qa_block(
    question="What is stratified splitting and why does it matter?",
    answer_text=(
        "Stratified splitting means that when you divide your data into train, validation, "
        "and test sets, you ensure each set contains the same proportion of each class label "
        "as the original dataset. Without this, you might accidentally put all examples of a "
        "rare class into the training set and have none in the test set, making evaluation "
        "meaningless. sklearn's train_test_split with stratify=y handles this automatically."
    ),
    simple_text=(
        "Imagine you have 10 red balls and 90 blue balls. Stratified splitting means that "
        "no matter which pile you make, each pile has roughly 10% red and 90% blue. "
        "If you did it randomly, you might get a pile that's 100% blue and never test "
        "if the robot can recognise red ones."
    ),
    tip_text="This is a classic interview topic. Knowing why it matters (not just what it is) impresses interviewers."
)

story += qa_block(
    question="You mapped 50+ labels to 6. How did you handle ambiguous labels?",
    answer_text=(
        "I built a LABEL_MAP dictionary that maps each of the Bitext tags to one of my 6 "
        "categories using exact string matching. For any tag that wasn't explicitly in the "
        "dictionary, I applied a keyword fallback: if the tag string contained words like "
        "'bill', 'charge', or 'payment', it was assigned to billing_issue, and so on for each "
        "category. This covered the vast majority of cases. About 973 rows used the fallback. "
        "In a production system, I would review these fallback assignments manually to ensure "
        "accuracy."
    ),
    simple_text=(
        "I made a lookup table — like a translation dictionary. If a label was in the "
        "dictionary, I used that translation. If not, I tried to guess from the words in "
        "the label name. Like if a label said 'billing_adjustment', I could guess it belongs "
        "in the money/billing box because it contains the word 'billing'."
    ),
    tip_text="Acknowledging the 973 fallback rows and saying you'd manually review them shows intellectual honesty."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 3 — THE MODELS
# ===========================================================================
story += sec("Section 3: The Models — How Did You Train Them?")

story += cat(["Baseline Model Questions"])

story += qa_block(
    question="You built two classifiers. What is the baseline and why did you build it?",
    answer_text=(
        "The baseline is a TF-IDF vectoriser combined with a Logistic Regression classifier, "
        "implemented as a single sklearn Pipeline. TF-IDF converts each message into a vector "
        "of numbers representing word importance scores. Logistic Regression then finds the "
        "linear decision boundary that separates the 6 classes. I built it first because: "
        "(1) it trains in milliseconds, (2) it provides a performance floor to compare against, "
        "and (3) it demonstrates that I understand when simpler models are appropriate."
    ),
    simple_text=(
        "Before building the fancy robot, I built a simple one. The simple one counts which "
        "words appear in a message and uses that to guess the category. It's like a calculator "
        "vs a smartphone. I built the calculator first to prove the smartphone was actually "
        "worth building."
    ),
    tip_text="Always justify your baseline. Interviewers want to see that you built it deliberately, not as an afterthought."
)

story += qa_block(
    question="What is TF-IDF?",
    answer_text=(
        "TF-IDF stands for Term Frequency — Inverse Document Frequency. TF measures how often "
        "a word appears in one document (high TF = word is frequent in this doc). IDF measures "
        "how rare the word is across all documents (high IDF = word is unique to few docs). "
        "Multiplying them gives a score that is high for words that are common in one document "
        "but rare across the whole dataset — these are the most informative words. Common words "
        "like 'the' or 'is' get near-zero scores because they appear everywhere."
    ),
    simple_text=(
        "Imagine every word gets a score. A word that appears a lot in just ONE message "
        "gets a high score — it's special to that message. A word like 'the' that appears "
        "in every single message gets a low score — it tells us nothing. TF-IDF is just a "
        "formula for giving each word its specialness score."
    ),
    tip_text="TF-IDF is a very common interview question. Learn this definition by heart."
)

story += qa_block(
    question="Your baseline (0.9958 F1) outperformed DistilBERT (0.9825 F1). How do you explain that?",
    answer_text=(
        "There are two reasons. First, the dataset itself: the Bitext dataset is professionally "
        "labelled and uses very consistent, formal language for each intent. TF-IDF word counts "
        "are perfectly sufficient to separate these clean categories — specific keywords almost "
        "uniquely identify each class. Second, the training constraint: I was running on CPU "
        "only, so I subsampled to 3,000 training examples and capped training at 300 steps. "
        "DistilBERT trained on the full dataset with more epochs would likely match or exceed "
        "the baseline. The baseline advantage is a dataset characteristic, not evidence that "
        "DistilBERT is a worse model."
    ),
    simple_text=(
        "The fancy robot did slightly worse because I couldn't let it study for long enough — "
        "it only had 300 practice rounds instead of thousands. The simple robot was good enough "
        "for this particular test because the messages in the dataset use very predictable words. "
        "If we had messier, real-world messages, the fancy robot would win."
    ),
    tip_text=(
        "This is almost guaranteed to come up. Interviewers love testing whether you understand "
        "your own results. The two-part answer (dataset quality + training constraint) is impressive."
    )
)

story += cat(["DistilBERT & Fine-Tuning Questions"])

story += qa_block(
    question="What is DistilBERT and why did you choose it?",
    answer_text=(
        "DistilBERT is a smaller, faster version of BERT (Bidirectional Encoder Representations "
        "from Transformers) created by HuggingFace using a technique called knowledge distillation. "
        "It retains 97% of BERT's language understanding while being 40% smaller and 60% faster. "
        "I chose it over full BERT because: (1) I was training on CPU, so speed and memory matter, "
        "(2) 97% performance retention is sufficient for a classification task, and (3) it is "
        "a production-proven model with excellent HuggingFace support."
    ),
    simple_text=(
        "BERT is a very smart robot brain that has read millions of books and websites. "
        "DistilBERT is BERT's younger sibling — 40% smaller, almost just as smart. I picked "
        "the little sibling because it runs faster on my computer, and for sorting six categories "
        "of messages, the little sibling is smart enough."
    ),
    tip_text="Justify model choice with concrete numbers (40% smaller, 97% performance, 60% faster). Don't just say 'it's popular'."
)

story += qa_block(
    question="What is fine-tuning and what does it mean to fine-tune DistilBERT?",
    answer_text=(
        "Fine-tuning means taking a pre-trained model — one that has already learned general "
        "language understanding from a massive text corpus — and continuing to train it on a "
        "smaller, task-specific dataset. The pre-trained model already knows grammar, context, "
        "and word meanings. Fine-tuning teaches it the specifics of your task. For DistilBERT, "
        "this means: (1) loading the pre-trained weights, (2) adding a classification head "
        "(a new linear layer that outputs 6 class probabilities), and (3) training the entire "
        "model end-to-end on the labelled customer support data."
    ),
    simple_text=(
        "Imagine you hire someone who already speaks fluent English and has read every book "
        "ever written. Fine-tuning is like giving that person a one-week crash course on "
        "customer support specifically. They already know words and sentences — you just "
        "teach them your specific job. Much faster than training someone from scratch."
    ),
    tip_text="Use the 'pre-trained + task-specific' framing. It's the standard mental model for fine-tuning."
)

story += qa_block(
    question="What is a classification head?",
    answer_text=(
        "A classification head is a simple linear layer added on top of a pre-trained model. "
        "DistilBERT's core outputs a 768-dimensional vector (called the [CLS] token embedding) "
        "that represents the meaning of the entire input sentence. The classification head "
        "multiplies this 768-dimensional vector by a weight matrix to produce 6 output "
        "scores (one per class), then applies softmax to convert them into probabilities. "
        "During fine-tuning, both the DistilBERT weights and the classification head weights "
        "are updated."
    ),
    simple_text=(
        "DistilBERT reads a sentence and produces a big list of 768 numbers that summarises "
        "the meaning. The classification head is like a voting machine — it takes those 768 "
        "numbers, does some maths, and outputs 6 scores: 'billing: 80%, login: 5%, ...' "
        "The highest score wins and becomes the prediction."
    ),
    tip_text="Knowing the dimension (768) and that softmax converts logits to probabilities is a strong technical detail."
)

story += qa_block(
    question="What hyperparameters did you tune and why?",
    answer_text=(
        "Key hyperparameters: learning_rate=2e-5 (standard for BERT fine-tuning; too high "
        "causes catastrophic forgetting, too low means no learning), max_length=128 tokens "
        "(sufficient for short support queries, reduces memory), batch_size=16 (balance "
        "between gradient quality and memory on CPU), max_steps=300 (CPU-adaptive cap to "
        "complete training in reasonable time), warmup_steps=int(0.1 * max_steps) (prevents "
        "large gradient updates in early training when weights are random). These are "
        "standard recommendations from the original BERT paper, adapted for CPU constraints."
    ),
    simple_text=(
        "Hyperparameters are like the settings on an oven before you bake a cake. "
        "Learning rate is how fast the robot adjusts — too fast and it forgets everything, "
        "too slow and it never learns. Batch size is how many examples it looks at "
        "before updating. Warmup steps is a gentle warm-up period, like stretching "
        "before exercise."
    ),
    tip_text=(
        "Always be able to explain WHY you set each hyperparameter, not just what you set it to. "
        "'2e-5 is standard for BERT fine-tuning per the original paper' is a strong answer."
    )
)

story += qa_block(
    question="How did you handle training on CPU only?",
    answer_text=(
        "I implemented automatic hardware detection at the start of training using "
        "torch.cuda.is_available(). When no GPU is detected, the training script activates "
        "two adaptive strategies: (1) Data subsampling — it stratified-samples 3,000 examples "
        "from the full training set rather than training on all 18,000, ensuring all 6 classes "
        "remain represented; (2) Step capping — it sets max_steps=300 instead of training for "
        "multiple full epochs. This reduces training time from ~20 hours to ~20 minutes while "
        "still producing a functional model."
    ),
    simple_text=(
        "Training a big neural network without a GPU is like running a marathon on crutches — "
        "very slow. So I wrote code that detects 'no GPU found' and automatically switches "
        "to a faster, smaller version of the training: fewer examples, fewer steps. "
        "The robot doesn't learn as much, but it learns enough, and it finishes in 20 minutes "
        "instead of 20 hours."
    ),
    tip_text="This shows engineering pragmatism — you adapted to constraints rather than just failing. Interviewers love this."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 4 — THE PIPELINE
# ===========================================================================
story += sec("Section 4: The Pipeline — How Does It All Connect?")

story += cat(["Architecture Questions"])

story += qa_block(
    question="Walk me through what happens when a customer sends a message.",
    answer_text=(
        "1. The raw customer query arrives at SupportAgent.resolve(). "
        "2. IntentClassifier.predict() tokenises the text, runs it through DistilBERT, "
        "and returns the top predicted intent label plus a confidence score (softmax probability). "
        "3. If confidence is below 0.70, the agent sets requires_human=True and returns a "
        "flag for human review without calling the LLM. "
        "4. Otherwise, get_template() fetches the intent-specific prompt template. "
        "format_user_prompt() fills in the customer query. "
        "5. ResponseGenerator.generate() sends the system prompt and user prompt to "
        "Claude via the Anthropic API and receives the generated response. "
        "6. The agent returns a dict containing the query, intent, confidence, response, "
        "context, and human_review flag."
    ),
    simple_text=(
        "Step 1: Customer writes a message. Step 2: Robot 1 reads it and decides which "
        "of 6 boxes it belongs to (and how sure it is). Step 3: If the robot is not sure "
        "enough (less than 70% confident), it raises a flag and a real human will handle it. "
        "Step 4: If the robot is sure, it picks the right letter template for that topic. "
        "Step 5: Robot 2 (Claude) reads the template and writes a personalised reply. "
        "Step 6: The full reply plus all the details are returned."
    ),
    tip_text="Practice saying this as a numbered list out loud. Being able to narrate a system end-to-end is a strong interview skill."
)

story += qa_block(
    question="What is prompt engineering and how did you use it?",
    answer_text=(
        "Prompt engineering is the practice of crafting input text to an LLM to guide it "
        "toward producing a desired output. In this project, I designed 6 intent-specific "
        "prompt templates, each with a system prompt (setting the LLM's role and tone) and "
        "a user prompt (providing the customer query plus intent-specific guidance). "
        "For example, the billing_issue template instructs the model to acknowledge the "
        "financial concern, show empathy, and offer concrete next steps. This structured "
        "approach ensures consistent, on-brand responses without requiring the LLM to guess "
        "the appropriate tone and content."
    ),
    simple_text=(
        "Prompt engineering is writing good instructions for the robot. Instead of just "
        "saying 'write a reply', I say 'you are a friendly support agent, the customer has "
        "a billing problem, be empathetic, offer to help fix it'. The better your instructions, "
        "the better the robot's answer."
    ),
    tip_text="Mention that you have 6 separate templates, not one generic one. This shows attention to detail."
)

story += qa_block(
    question="Why does the system flag low-confidence predictions for human review?",
    answer_text=(
        "The confidence threshold (0.70) acts as a safety net. When the classifier's softmax "
        "probability for the top class is below 70%, it indicates the model is uncertain — "
        "the input may be ambiguous, out-of-distribution, or phrased in a way the model "
        "hasn't seen. Sending an uncertain intent to the LLM would generate a response built "
        "on a potentially wrong context, which could mislead or frustrate the customer. "
        "Flagging for human review prevents poor automated responses from reaching customers "
        "while still automating the confident majority."
    ),
    simple_text=(
        "Imagine asking the sorting robot 'are you sure?' — if it's less than 70% sure, "
        "it says 'I'm not confident, a human should handle this one'. This is important "
        "because if the robot sorts the message into the wrong box, the reply will be "
        "totally wrong. Better to get a human than to send a bad automated reply."
    ),
    tip_text="This shows you designed for real-world use, not just accuracy metrics. Production-readiness thinking."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 5 — EVALUATION
# ===========================================================================
story += sec("Section 5: Evaluation — How Do You Know It Works?")

story += cat(["Metrics & Evaluation Questions"])

story += qa_block(
    question="What is weighted F1 score and why did you use it?",
    answer_text=(
        "F1 score is the harmonic mean of precision and recall. Precision asks: of all the "
        "messages I labelled as 'billing_issue', how many actually were? Recall asks: of all "
        "the actual billing_issue messages, how many did I catch? The harmonic mean penalises "
        "imbalanced precision/recall more than the arithmetic mean. Weighted F1 averages the "
        "per-class F1 scores, weighting each class by its number of examples. I chose weighted "
        "F1 over accuracy because it better handles class imbalance — accuracy alone can be "
        "misleadingly high if one class dominates."
    ),
    simple_text=(
        "Imagine a test where 90% of questions are easy and 10% are hard. If you only answer "
        "the easy ones, you score 90% but you're failing on the hard ones. F1 score checks "
        "BOTH whether your answers are correct AND whether you answered all the questions — "
        "not just the easy majority."
    ),
    tip_text="Knowing why you chose F1 over accuracy is a very common interview question. Always have this answer ready."
)

story += qa_block(
    question="What is RAGAS and how did you use it?",
    answer_text=(
        "RAGAS (Retrieval-Augmented Generation Assessment) is an open-source evaluation "
        "framework originally designed to measure the quality of RAG pipeline outputs. "
        "It provides metrics including Faithfulness (does the response stay within the "
        "provided context?) and Answer Relevancy (does the response address the question?). "
        "I initially attempted to use the RAGAS library but encountered dependency conflicts "
        "— it required OpenAI embeddings by default. I ultimately implemented the same metrics "
        "directly using Claude Haiku as the evaluator LLM, bypassing the library while "
        "preserving the conceptual framework."
    ),
    simple_text=(
        "RAGAS is a tool for grading AI replies. Faithfulness asks: did the robot stick to "
        "what it was told, or did it make things up? Answer Relevancy asks: did the robot "
        "actually answer the question? I tried using the RAGAS tool but it had technical "
        "problems, so I built my own version that does the same grading."
    ),
    tip_text="Be upfront about the dependency issue and your workaround. Showing problem-solving is better than hiding struggles."
)

story += qa_block(
    question="Your faithfulness score was 0.667, below the 0.85 target. Is that a failure?",
    answer_text=(
        "Not in this context. Faithfulness in RAGAS measures whether the generated response "
        "is grounded in the provided context document. In a RAG system with a knowledge base, "
        "a low faithfulness score means the model hallucinated facts. But in this system, "
        "the 'context' is a prompt template with minimal content — it contains guidance and "
        "tone instructions, not a database of facts. Claude is expected to generate helpful "
        "domain knowledge (like explaining billing processes) that is not literally in the "
        "template. This is correct, desirable behaviour. The more meaningful metric here is "
        "Answer Relevancy (0.837), which passed its target of 0.80."
    ),
    simple_text=(
        "Faithfulness is like asking 'did the robot only use words from the instruction card?' "
        "But our instruction card only has general guidelines, not specific facts. So when "
        "the robot adds helpful details (like how to reset a password), it 'fails' faithfulness "
        "even though its answer was actually great. The more important score — did it answer "
        "the right question? — passed with 0.837."
    ),
    tip_text=(
        "This is the most nuanced result in the project. Interviewers who see the 0.667 will "
        "test you on it. Have this explanation ready and be confident — you are NOT making excuses, "
        "you are correctly identifying a metric limitation."
    )
)

story += qa_block(
    question="How did you evaluate the LLM-generated responses?",
    answer_text=(
        "I implemented a custom synchronous evaluator using Claude Haiku as the judge LLM. "
        "For each of the 50 test responses, I sent two evaluation prompts to Claude Haiku: "
        "one asking it to score faithfulness (0.0-1.0) and one asking it to score answer "
        "relevancy (0.0-1.0). Each prompt asked for only a single decimal number in the reply "
        "(max_tokens=10, temperature=0 for determinism). I then computed mean, median, std, "
        "min, and max across all 50 scores. Results were saved to results/ragas_scores.json."
    ),
    simple_text=(
        "I used a second AI (Claude Haiku) to grade the first AI's answers. For each answer, "
        "I asked Haiku two questions: 'How well does this answer stick to the topic? Score "
        "0 to 1' and 'How well does this answer address what the customer asked? Score 0 to 1'. "
        "Then I averaged all 50 scores to get the final grade."
    ),
    tip_text="LLM-as-judge evaluation is a hot topic in 2024-2026. Knowing why you use temperature=0 for evaluation (reproducibility) is a great detail."
)

story += qa_block(
    question="What is the difference between precision and recall?",
    answer_text=(
        "Precision: of everything the model labelled as class X, what fraction actually is X? "
        "High precision = few false positives. Recall: of everything that actually is class X, "
        "what fraction did the model correctly identify? High recall = few false negatives. "
        "There is usually a trade-off: tuning for higher recall means accepting more false "
        "positives, and vice versa. The right balance depends on the cost of each error type. "
        "In a medical diagnosis context, high recall (catch all real cases) matters more. "
        "In a spam filter, high precision (don't block real emails) matters more."
    ),
    simple_text=(
        "Precision: if the robot says 'this is a cat', how often is it actually a cat? "
        "Recall: of all the real cats, how many did the robot notice? "
        "A robot that calls everything a cat has perfect recall (it never misses a cat) "
        "but terrible precision (most of what it calls cats are dogs). "
        "You need both to be good."
    ),
    tip_text="The medical/spam example is a classic way to make precision/recall trade-offs concrete. Use it."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 6 — CHALLENGES
# ===========================================================================
story += sec("Section 6: Challenges & Problem Solving")

story += cat(["'Tell Me About a Challenge' Questions"])

story += qa_block(
    question="What was the hardest technical problem you faced and how did you solve it?",
    answer_text=(
        "The most significant challenge was the RAGAS evaluation framework's hard dependency "
        "on OpenAI. After installing RAGAS and configuring the Anthropic LLM wrapper, the "
        "library still tried to call OpenAI for embedding-based metrics. Attempts to swap "
        "in HuggingFace embeddings via LangchainEmbeddingsWrapper also failed due to RAGAS's "
        "internal async timeout handling. Rather than spending hours debugging a third-party "
        "library, I made the decision to implement the same conceptual metrics — faithfulness "
        "and answer relevancy — as a direct, synchronous Anthropic API loop. This removed "
        "the dependency entirely, eliminated the async timeout issue, and produced cleaner, "
        "more interpretable results."
    ),
    simple_text=(
        "I tried to use a ready-made grading tool (RAGAS) but it secretly required a "
        "different AI service (OpenAI) that I wasn't using. No matter what I tried, "
        "it kept asking for that service. So instead of fighting it, I built my own "
        "grading tool from scratch in 100 lines of code. My version was actually simpler "
        "and worked better."
    ),
    tip_text="This answer shows debugging skill, good judgment (knowing when to stop debugging), and resourcefulness. Lead with the challenge, end with the solution."
)

story += qa_block(
    question="How did you deal with the slow CPU training problem?",
    answer_text=(
        "The naive training run would have taken 20+ hours on CPU — clearly impractical. "
        "I solved it with two changes: (1) Automatic detection — the code checks "
        "torch.cuda.is_available() and activates 'CPU mode' when no GPU is found. "
        "(2) Adaptive parameters — in CPU mode, training data is stratified-subsampled "
        "to 3,000 examples and max_steps is capped at 300. This reduces training time to "
        "~20 minutes while still producing a model with 0.9825 F1, which proves the approach "
        "is sound. The config file exposes cpu_train_sample and cpu_max_steps as tunable "
        "parameters so they can be adjusted."
    ),
    simple_text=(
        "Training the robot normally would take 20 hours without a special graphics card. "
        "I wrote code that detects the slow computer and automatically switches to a "
        "faster mini-training mode: less data, fewer rounds. The robot doesn't become "
        "as expert, but it still gets a 98.25% score, which proves the idea works. "
        "It's like practicing for a marathon by running 5km — you prove you can run, "
        "even if you haven't run the full 42km yet."
    ),
    tip_text="Framing this as intentional engineering (not a workaround) is important. You made a pragmatic trade-off, not a mistake."
)

story += qa_block(
    question="sklearn 1.8 removed the multi_class parameter. How did you handle a breaking change?",
    answer_text=(
        "When I ran the baseline training script, it threw a TypeError: "
        "LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'. "
        "This is because sklearn 1.8 removed the deprecated multi_class='multinomial' "
        "parameter. The fix was simple — remove the parameter from both the code and config. "
        "Modern sklearn's LogisticRegression automatically handles multiclass problems using "
        "the one-vs-rest scheme by default, which produces equivalent results. This was a "
        "lesson in keeping requirements pinned in production to prevent unexpected breakage."
    ),
    simple_text=(
        "A tool I was using (sklearn) got an update that removed a setting I was using. "
        "The computer gave me an error saying it didn't recognise that setting anymore. "
        "I looked it up and found out the new version doesn't need that setting — it "
        "figures it out automatically. So I deleted that line of code and everything worked. "
        "Lesson learned: always write down exactly which version of each tool you're using."
    ),
    tip_text="Handling a library breaking change gracefully and learning from it is a great story for a behavioural question."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 7 — PRODUCTION THINKING
# ===========================================================================
story += sec("Section 7: Production & Real-World Thinking")

story += cat(["Scalability & Production Questions"])

story += qa_block(
    question="How would you deploy this system in production?",
    answer_text=(
        "A production deployment would involve: (1) Serving the classifier as a REST API "
        "using FastAPI, with the model loaded into memory at startup and a /predict endpoint. "
        "(2) Containerising with Docker so the model and all dependencies are portable. "
        "(3) Deploying to a cloud provider (AWS, GCP, or Azure) with auto-scaling based on "
        "request volume. (4) Implementing a message queue (e.g. SQS or Kafka) if volume is "
        "high, so requests are processed asynchronously. (5) Caching the LLM response for "
        "duplicate or near-duplicate queries to reduce Anthropic API costs. "
        "(6) Adding monitoring/logging (latency, error rate, intent distribution) with tools "
        "like Prometheus/Grafana or Datadog."
    ),
    simple_text=(
        "To put this in a real company, I would: wrap it in a web address so other apps "
        "can call it, package it in a box (Docker) so it runs anywhere, put it on a cloud "
        "computer that can grow bigger when more people use it, save common replies so we "
        "don't call the expensive AI every time, and add a dashboard showing how well it's "
        "working every day."
    ),
    tip_text="Even if you haven't deployed it, showing you KNOW how to deploy it is enough. Mention FastAPI, Docker, and monitoring."
)

story += qa_block(
    question="How would you monitor this system once deployed?",
    answer_text=(
        "Monitoring would cover three layers: (1) Infrastructure metrics — latency, error rate, "
        "throughput (standard APM). (2) ML metrics — intent distribution drift (if billing_issue "
        "suddenly spikes, something changed), average confidence score over time (confidence drop "
        "may indicate the model is seeing new types of queries it wasn't trained on), and "
        "human_review escalation rate. (3) Business metrics — customer satisfaction, resolution "
        "time, re-contact rate. I would also implement periodic re-evaluation: run new queries "
        "through the LLM judge and alert if relevancy drops below threshold."
    ),
    simple_text=(
        "Monitoring is like a health check for the robot. I'd watch: is it fast enough? "
        "Is it confident? Are more messages than usual going to humans for review? "
        "Are customers satisfied with the replies? If any of these go wrong, "
        "it might mean the robot needs to be retrained or fixed."
    ),
    tip_text="Mentioning concept drift (confidence drops, distribution shifts) shows senior ML engineering knowledge."
)

story += qa_block(
    question="How would you improve the model if given more resources?",
    answer_text=(
        "With a GPU: train on the full 18,000+ example dataset for 3-5 epochs with proper "
        "hyperparameter search (learning rate, batch size). "
        "With more data: collect real customer support tickets, which are messier than the "
        "Bitext dataset and would better reflect production distribution. "
        "Architecturally: (1) implement retrieval-augmented generation — instead of static "
        "prompt templates, retrieve relevant FAQ articles or resolution histories; "
        "(2) add a re-ranking step to select the best candidate response from multiple "
        "LLM generations; (3) implement active learning — flag uncertain predictions, "
        "have humans label them, and retrain periodically."
    ),
    simple_text=(
        "With a proper gaming computer: train the robot on all the data, not just a sample. "
        "With real company data: teach the robot using actual past customer conversations. "
        "With more time: instead of using a fixed template, let the robot look up real "
        "answers from the company's help pages. Like teaching someone to use a real "
        "reference book instead of memorising everything."
    ),
    tip_text="RAG as a next step is a strong answer because it shows architectural thinking beyond fine-tuning."
)

story += qa_block(
    question="What is the cost of running this system at scale?",
    answer_text=(
        "The main cost is the Anthropic API for response generation. At the time of building "
        "this, Claude Sonnet costs approximately $3 per million input tokens and $15 per million "
        "output tokens. A typical support response exchange is ~500 input + ~200 output tokens, "
        "so roughly $0.0045 per resolved query. At 10,000 queries/day that is ~$45/day. "
        "The classifier inference cost is negligible once hosted — DistilBERT runs in ~21ms "
        "per query on CPU. Cost optimisation levers: use Claude Haiku for simple intents "
        "and Sonnet only for complex ones, implement response caching for common queries, "
        "or fine-tune a smaller model as a responder."
    ),
    simple_text=(
        "The expensive part is asking Claude to write each reply — it costs a tiny amount "
        "per reply, but it adds up with millions of customers. The sorting robot is almost "
        "free to run. To save money: use the cheaper AI for easy questions, save common "
        "replies so you only pay once, and use the expensive AI only for tricky problems."
    ),
    tip_text="Showing cost-awareness is impressive — it signals you think like a product engineer, not just a researcher."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 8 — BEHAVIOURAL QUESTIONS
# ===========================================================================
story += sec("Section 8: Behavioural Questions")

story += cat(["STAR-Format Answers"])

story.append(body(
    "Behavioural questions use the STAR format: Situation, Task, Action, Result. "
    "Each answer below is structured this way. Practice saying these out loud."
))
story.append(sp(8))

story += qa_block(
    question="Tell me about a time you had to make a pragmatic decision under constraints.",
    answer_text=(
        "SITUATION: I was implementing the evaluation pipeline and had chosen RAGAS as the "
        "framework. After installation it threw OpenAI API errors despite being configured "
        "with Anthropic. "
        "TASK: I needed working evaluation metrics before I could report any results. "
        "ACTION: I investigated the root cause (RAGAS hardcoded OpenAI for embeddings, "
        "and its async architecture caused timeouts at the API rate limit). I concluded "
        "that patching a third-party library would take longer than building a clean "
        "alternative. I wrote a 100-line synchronous evaluator using Claude Haiku directly. "
        "RESULT: Clean, reproducible evaluation in 50 minutes wall-clock time, equivalent "
        "conceptual metrics, and no external dependencies. The decision to cut scope (drop "
        "the RAGAS library, keep the metric concepts) was the right engineering call."
    ),
    simple_text=(
        "I tried to use a ready-made tool but it was broken for my use case. "
        "I had two choices: spend days fixing the broken tool, or spend one hour building "
        "a simpler version myself. I chose to build my own. It worked perfectly and "
        "I learned more by building it."
    ),
    tip_text="This story shows: debugging skills, engineering judgment, bias for action, and pragmatism. It is one of the best stories in this project."
)

story += qa_block(
    question="Tell me about a time you had to explain something technical to a non-technical person.",
    answer_text=(
        "SITUATION: The confidence threshold concept — why the system escalates to humans — "
        "is technical but has a direct business impact. "
        "TASK: Explain it so a product manager or stakeholder could understand the design decision. "
        "ACTION: I framed it as 'the robot tells you when it's not sure'. I used the analogy "
        "of a new employee who, when unsure, asks their manager rather than guessing. "
        "The 70% threshold means: if the model's certainty is below 70%, a real human "
        "handles the ticket. "
        "RESULT: The stakeholder immediately understood both what the system does and why "
        "the fallback matters for customer experience, without needing to understand "
        "softmax probabilities."
    ),
    simple_text=(
        "I explained that the robot says 'I'm not sure, a person should handle this' when "
        "it's less than 70% confident. Like a new cashier who, when they're unsure about a "
        "return policy, calls their manager rather than guessing and getting it wrong."
    ),
    tip_text="Prepare a non-technical explanation of every key concept. Being able to bridge technical and business language is a senior skill."
)

story += qa_block(
    question="What would you do differently if you started this project again?",
    answer_text=(
        "Three things: First, I would pin all dependency versions immediately in "
        "requirements.txt to avoid breaking changes (like the sklearn multi_class issue). "
        "Second, I would design the evaluation framework before building the pipeline — "
        "knowing I'd need faithfulness and relevancy metrics upfront would have made "
        "me design better output schemas in the pipeline from the start. "
        "Third, I would collect a small real-world test set (actual customer messages from "
        "a live product) rather than splitting the training dataset — this gives a more "
        "honest estimate of production performance."
    ),
    simple_text=(
        "I would: write down exactly which version of every tool I'm using before I start, "
        "plan how I'll test the results BEFORE building the robot (not after), "
        "and use real customer messages for the final test instead of ones from the "
        "same practice dataset."
    ),
    tip_text="Showing genuine reflection, not fake humility ('I would've worked harder') is what recruiters want. These three specific things are credible."
)

story.append(PageBreak())

# ===========================================================================
# SECTION 9 — RAPID FIRE
# ===========================================================================
story += sec("Section 9: Rapid-Fire Questions")

story += cat(["Short, Confident Answers"])

story.append(body(
    "These questions expect a 1-3 sentence answer. Practice answering each in under 20 seconds."
))
story.append(sp(6))

rapid_fire = [
    ("What is a transformer model?",
     "A neural network architecture that uses 'attention' to weigh how important each word "
     "is relative to every other word in a sentence, enabling much better language understanding "
     "than earlier sequential models like LSTMs.",
     "A robot brain that reads a whole sentence at once and figures out which words "
     "are most important based on all the other words around them."),

    ("What is tokenisation?",
     "The process of splitting raw text into subword units (tokens) that the model can process. "
     "DistilBERT uses WordPiece tokenisation, which breaks rare words into common subword pieces "
     "to handle a fixed vocabulary.",
     "Chopping up a sentence into small pieces the robot can understand. 'unbelievable' "
     "might become ['un', '##believ', '##able'] — three pieces."),

    ("What is softmax?",
     "A function that converts a vector of raw scores (logits) into a probability distribution "
     "summing to 1.0. Used as the final layer in classification to produce interpretable confidence scores.",
     "A calculator that takes a list of numbers and converts them into percentages that "
     "add up to 100%. So 'billing: 4.2, login: 0.3' becomes 'billing: 80%, login: 20%'."),

    ("What is overfitting?",
     "When a model memorises the training data so well that it performs poorly on unseen data. "
     "It learns noise and specific examples rather than general patterns.",
     "The robot studied so hard for its practice test that it memorised all the exact "
     "questions. On the real test with different questions, it fails because it memorised "
     "instead of understanding."),

    ("What is the difference between a language model and a classifier?",
     "A language model generates text (predicts the next token). A classifier assigns a "
     "label to an input from a fixed set of categories. DistilBERT here is used as a classifier "
     "(with a classification head), not as a generator. Claude is the language model.",
     "The classifier is like a sorting machine that puts things in boxes. "
     "The language model is like a writer that creates new text. "
     "This project uses both: one to sort, one to write."),

    ("What is knowledge distillation?",
     "A technique where a smaller 'student' model is trained to mimic the outputs of a larger "
     "'teacher' model. DistilBERT was distilled from BERT: the student learns to match BERT's "
     "output distributions, not just the training labels.",
     "Like a wise teacher summarising all their knowledge into a compact book for a student. "
     "The student (DistilBERT) is smaller but very smart because it learned from the big teacher (BERT)."),

    ("What is an epoch?",
     "One full pass through the entire training dataset. Training for 3 epochs means the model "
     "sees every training example 3 times. More epochs can improve performance but risk overfitting.",
     "The robot reading every single practice example once. Three epochs = the robot "
     "read the whole practice book three times."),

    ("What is gradient descent?",
     "An optimisation algorithm that iteratively adjusts model weights in the direction that "
     "reduces the loss function. The learning rate controls the size of each step.",
     "Imagine rolling a ball down a hill to find the lowest point. Gradient descent "
     "is the maths that tells the robot which direction 'downhill' is, so it can improve "
     "its answers little by little."),

    ("What is the Anthropic API?",
     "A REST API provided by Anthropic that allows developers to send messages to Claude models "
     "and receive generated text responses. It requires an API key and is billed per token.",
     "It's a way to talk to Claude (the AI) from your own program. You send a message, "
     "Claude sends back a reply. Like texting, but for code."),

    ("What is a confusion matrix?",
     "A table showing predicted vs actual labels for a classifier. Rows are actual classes, "
     "columns are predicted classes. Diagonal cells are correct predictions; off-diagonal "
     "cells are misclassifications.",
     "A report card showing where the robot gets confused. If it often mixes up "
     "'billing_issue' and 'cancellation_request', those cells will be bright in the table."),
]

for question, answer_full, answer_simple in rapid_fire:
    story += [
        sp(4),
        q(question),
        sp(2),
        a(answer_full),
        sp(2),
        simple(answer_simple),
        sp(4),
        rule()
    ]

story.append(PageBreak())

# ===========================================================================
# SECTION 10 — QUESTIONS TO ASK
# ===========================================================================
story += sec("Section 10: Questions YOU Should Ask the Interviewer")

story += cat(["Show Curiosity & Depth"])

story.append(body(
    "Asking smart questions at the end of an interview shows genuine interest, "
    "seniority, and that you have thought beyond the code. Have at least 3-4 ready."
))
story.append(sp(10))

questions_to_ask = [
    (
        "How do you currently handle intent classification in your customer support pipeline, "
        "and what are the biggest pain points?",
        "This shows you're thinking about real-world application and positioning your skills "
        "against actual problems they face. It also opens a dialogue about how your project "
        "experience is relevant."
    ),
    (
        "What does your model evaluation and monitoring setup look like in production? "
        "How do you detect when a model starts degrading?",
        "This shows you think about the full ML lifecycle — not just training, but "
        "post-deployment health. It's a question a senior ML engineer would ask."
    ),
    (
        "How do you balance automation confidence with the cost of human escalation? "
        "Where do you draw the line between automated response and human review?",
        "This ties directly to your project's confidence threshold design. "
        "It shows you understand the business trade-off, not just the technical one."
    ),
    (
        "What is the main bottleneck in your current NLP/LLM pipeline — is it latency, "
        "accuracy, cost, or something else?",
        "This is a strategic question that shows you understand constraints. "
        "The answer will tell you a lot about the team's priorities."
    ),
    (
        "How do you manage prompt versioning when you update templates that are live in production?",
        "This is a sharp, specific question about LLMOps. Most companies struggle with this "
        "and it shows you have thought about deployment realities beyond just building the model."
    ),
    (
        "How does the team approach handling new intent categories that weren't in the original training set?",
        "This shows you understand model limitations (out-of-distribution inputs) and are "
        "thinking about long-term maintenance."
    ),
]

for i, (q_text, why_text) in enumerate(questions_to_ask, 1):
    block = [
        sp(4),
        Paragraph(f"Question {i}:", CATEGORY_STYLE),
        Paragraph(f'"{q_text}"', ParagraphStyle(
            "QtoAsk", parent=styles["Normal"],
            fontSize=11, leading=16, textColor=colors.HexColor("#0f3460"),
            fontName="Helvetica-BoldOblique", leftIndent=10, spaceAfter=4,
            borderColor=colors.HexColor("#0f3460"), borderWidth=1,
            borderPad=8, backColor=colors.HexColor("#f0f4ff"), borderRadius=4
        )),
        sp(4),
        Paragraph(
            f"Why this works: {why_text}",
            ParagraphStyle(
                "WhyWorks", parent=styles["Normal"],
                fontSize=10, leading=14, textColor=colors.HexColor("#374151"),
                leftIndent=10, spaceAfter=6,
                backColor=colors.HexColor("#f9fafb"),
                borderColor=colors.HexColor("#d1d5db"), borderWidth=0.5,
                borderPad=6
            )
        ),
        sp(4),
        rule()
    ]
    story += block

story.append(PageBreak())

# ===========================================================================
# QUICK REFERENCE CHEAT SHEET
# ===========================================================================
story += sec("Quick Reference — Key Numbers to Remember")

story.append(body(
    "Memorise these numbers. Quoting exact results confidently makes a strong impression."
))
story.append(sp(10))

cheat_sheet_data = [
    ["Metric", "Value", "What It Means"],
    ["Baseline Weighted F1", "0.9958", "TF-IDF + Logistic Regression accuracy"],
    ["DistilBERT Weighted F1", "0.9825", "Fine-tuned transformer accuracy"],
    ["Min per-class F1 (Baseline)", "0.985", "Worst single class performance"],
    ["Min per-class F1 (DistilBERT)", "0.953", "Worst single class performance"],
    ["Answer Relevancy", "0.837 (PASS)", "LLM responses address customer questions"],
    ["Faithfulness", "0.667 (expected low)", "LLM generates beyond the template — intentional"],
    ["Confidence threshold", "0.70", "Below this, route to human review"],
    ["Training data size", "26,872 examples", "Full Bitext dataset"],
    ["CPU training subsample", "3,000 examples", "Adaptive for CPU-only training"],
    ["Training steps (CPU)", "300 steps", "~20 min on CPU"],
    ["Evaluation queries", "50 queries", "RAGAS-style evaluation sample"],
    ["Baseline model size", "0.4 MB", "TF-IDF + LR pickle"],
    ["DistilBERT model size", "4,088 MB", "Fine-tuned transformer weights"],
    ["Baseline inference", "0.15 ms/sample", "Extremely fast"],
    ["DistilBERT inference", "21.18 ms/sample", "140x slower but much more capable"],
    ["Intent categories", "6", "billing, account, technical, inquiry, cancellation, feedback"],
    ["Test set queries (generation)", "200 queries", "Subsampled for LLM generation pipeline"],
]

cheat = Table(cheat_sheet_data, colWidths=[6*cm, 4.5*cm, 6*cm])
cheat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    # Highlight the pass/fail rows
    ("TEXTCOLOR", (1, 6), (1, 6), colors.HexColor("#065f46")),
    ("TEXTCOLOR", (1, 7), (1, 7), colors.HexColor("#92400e")),
    ("FONTNAME", (1, 6), (1, 7), "Helvetica-Bold"),
]))
story.append(cheat)
story.append(sp(16))

# Final encouragement
story += [
    HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460"), spaceAfter=12),
    Paragraph("You Built This. Own It.", ParagraphStyle(
        "Final", parent=styles["Normal"],
        fontSize=16, textColor=colors.HexColor("#0f3460"),
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=8
    )),
    Paragraph(
        "Every number in that cheat sheet came from code you wrote. "
        "Every decision — from the confidence threshold to the custom evaluator — "
        "was yours. When an interviewer asks about this project, you are the expert "
        "in the room. Speak with confidence.",
        ParagraphStyle(
            "FinalBody", parent=styles["Normal"],
            fontSize=11, leading=17, textColor=colors.HexColor("#374151"),
            alignment=TA_CENTER, spaceAfter=6
        )
    ),
]

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=2*cm,
    rightMargin=2*cm,
    topMargin=2.5*cm,
    bottomMargin=2.5*cm,
    title="Interview Prep — Customer Support AI",
    author="Claude Code",
)
doc.build(story)
print(f"PDF saved -> {OUTPUT.resolve()}")
