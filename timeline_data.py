"""
timeline_data.py — the AI Evolution Timeline, as data.

The Evolution of Machine Learning and AI · Week 1 Lab
South East Technological University

Sixty-odd milestones, each tagged with the two things the Week 1 lecture cares
about: which FAMILY it belongs to (symbolic or connectionist, the split from
Dartmouth 1956), and which INGREDIENT it supplied (data, compute, architecture,
or none of them).

Tagging the record this way turns the timeline from a list of dates into
something you can query. The lab's exercises are all queries:

    - plot the two families' fortunes against each other over 75 years
    - show that 2012 is the first year all three ingredients are present
    - find the milestones that arrived during a "dead" field

Booklet cross-references: Ch. 2-13 (the narrative), Ch. 15 (master timeline).

Fields
------
year        int
name        str
kind        'idea' | 'system' | 'dataset' | 'hardware' | 'institution' | 'shock'
family      'symbolic' | 'connectionist' | 'statistical' | 'neither'
ingredient  'data' | 'compute' | 'architecture' | None
note        one line, in the booklet's voice
"""

from __future__ import annotations

import pandas as pd

# (year, name, kind, family, ingredient, note)
MILESTONES = [
    (1843, "Lovelace's objection", "idea", "neither", None,
     "The machine can only do what we know how to order it to perform."),
    (1854, "Boole, Laws of Thought", "idea", "symbolic", None,
     "Reasoning is algebra."),
    (1936, "Turing machine", "idea", "neither", None,
     "One machine can run any program."),
    (1937, "Shannon's thesis", "idea", "neither", None,
     "That algebra is circuitry."),
    (1943, "McCulloch and Pitts neuron", "idea", "connectionist", "architecture",
     "Simplified neurons compute logic. Weights fixed by hand."),
    (1949, "Hebbian learning", "idea", "connectionist", None,
     "Cells that fire together wire together."),
    (1950, "Turing's Imitation Game", "idea", "neither", None,
     "Replaces 'can machines think?' with something measurable."),
    (1956, "Dartmouth workshop", "institution", "symbolic", None,
     "The field is named, and overdoses on optimism in the same document."),
    (1956, "Logic Theorist", "system", "symbolic", None,
     "Proves theorems from Principia; one proof more elegant than the published one."),
    (1957, "Simon and Newell forecast", "shock", "symbolic", None,
     "Chess champion within ten years. It took forty."),
    (1958, "Perceptron", "system", "connectionist", "architecture",
     "The first machine that learns from error. 400 pixels."),
    (1958, "LISP", "system", "symbolic", None,
     "The field's native tongue for thirty years."),
    (1959, "Samuel's checkers player", "system", "statistical", None,
     "Coins 'machine learning'; improves by self-play."),
    (1965, "DENDRAL begun", "system", "symbolic", None,
     "Infers molecular structure at expert level."),
    (1966, "ELIZA", "system", "symbolic", None,
     "Keyword to template. Understanding: zero. The fluency trap opens."),
    (1966, "ALPAC report", "shock", "neither", None,
     "Machine translation funding collapses. The first documented promise gap."),
    (1969, "Perceptrons (Minsky and Papert)", "shock", "symbolic", None,
     "XOR: correct about one layer, read as a refutation of the idea."),
    (1969, "Shakey the robot", "system", "symbolic", None,
     "'The first electronic person', in a prepared room."),
    (1970, "SHRDLU", "system", "symbolic", None,
     "Genuine understanding, of blocks."),
    (1972, "Prolog", "system", "symbolic", None,
     "Logic as a programming language."),
    (1973, "Lighthill report", "shock", "neither", None,
     "UK funding withdrawn. Winter I begins."),
    (1974, "WINTER I begins", "shock", "neither", None,
     "Combinatorial explosion meets promises that assumed it away."),
    (1975, "MYCIN evaluated", "system", "symbolic", None,
     "Matches specialist physicians. Never deployed: liability, workflow, trust."),
    (1979, "Neocognitron", "system", "connectionist", "architecture",
     "Convolutional structure, a decade before it could be trained."),
    (1980, "XCON at DEC", "system", "symbolic", None,
     "Configures orders; credited with about $40M a year saved."),
    (1980, "Expert systems boom", "institution", "symbolic", None,
     "A smaller, sellable promise. It works, for seven years."),
    (1982, "Fifth Generation programme", "institution", "symbolic", None,
     "Japan's national bet; the West panics and funds counter-programmes."),
    (1982, "Hopfield networks", "idea", "connectionist", "architecture",
     "Connectionism keeps working through the winter."),
    (1986, "Backpropagation popularised", "idea", "connectionist", "architecture",
     "XOR's escape route, seventeen years late. Still trains everything today."),
    (1987, "Lisp machine market collapses", "shock", "symbolic", None,
     "Commodity desktops cross over. A half-billion-dollar market evaporates."),
    (1987, "WINTER II begins", "shock", "neither", None,
     "Brittleness, the knowledge bottleneck, and maintenance present their invoices."),
    (1989, "LeNet / digit recognition", "system", "connectionist", "architecture",
     "Deep learning in production, reading cheques, a decade before it was famous."),
    (1990, "The great rebrand", "institution", "statistical", None,
     "'AI' becomes toxic; the work continues as 'machine learning'."),
    (1995, "Support vector machines", "idea", "statistical", None,
     "Statistics wins the 1990s. Neural networks are career poison."),
    (1997, "Deep Blue beats Kasparov", "system", "symbolic", "compute",
     "200 million positions a second. Zero transfer. The old paradigm's finest hour."),
    (1997, "LSTM", "idea", "connectionist", "architecture",
     "Long-range memory for sequences. Waits twenty years for its moment."),
    (1998, "Google PageRank", "system", "statistical", "data",
     "AI colonises daily life under another name."),
    (2001, "Statistical machine translation", "system", "statistical", "data",
     "More data beats better rules. The bitter lesson, early."),
    (2006, "Deep belief networks", "idea", "connectionist", "architecture",
     "Hinton shows deep nets can be trained. The term 'deep learning' returns."),
    (2007, "CUDA released", "hardware", "neither", "compute",
     "Gamers' silicon becomes programmable. The road is built for other reasons."),
    (2009, "ImageNet released", "dataset", "statistical", "data",
     "14M labelled images, built against advice that it was service work."),
    (2010, "ImageNet challenge begins", "institution", "statistical", "data",
     "An arena with no champion. Data and compute, no architecture yet."),
    (2011, "IBM Watson wins Jeopardy!", "system", "statistical", "data",
     "Statistical NLP at scale, still hand-engineered."),
    (2012, "AlexNet", "system", "connectionist", "architecture",
     "26% to 15% error. Data plus compute plus architecture, for the first time."),
    (2013, "word2vec", "idea", "connectionist", "architecture",
     "Meaning as geometry. Embeddings enter the toolkit."),
    (2014, "GANs", "idea", "connectionist", "architecture",
     "Forger against detective. Opens the provenance and deepfake agenda."),
    (2015, "ResNet", "idea", "connectionist", "architecture",
     "Skip connections make depth trainable. Surpasses human top-5 on ImageNet."),
    (2016, "AlphaGo beats Lee Sedol", "system", "connectionist", "compute",
     "Move 37. Learned intuition where brute force was permanently out."),
    (2017, "Transformer", "idea", "connectionist", "architecture",
     "Every word attends to every other, in parallel. The architecture the GPU was waiting for."),
    (2018, "BERT and GPT", "system", "connectionist", "data",
     "Pretrain then fine-tune. The recipe for the next five years."),
    (2018, "Turing Award to Hinton, LeCun, Bengio", "institution", "connectionist", None,
     "The keepers of the flame, thirty years later."),
    (2020, "Scaling laws", "idea", "connectionist", "compute",
     "Error falls smoothly with compute. Capability becomes purchasable."),
    (2020, "GPT-3", "system", "connectionist", "data",
     "Emergence: abilities nobody engineered, ahead of our understanding of why."),
    (2020, "AlphaFold 2", "system", "connectionist", "architecture",
     "Structure prediction at experimental accuracy. Science-grounded AI arrives."),
    (2022, "ChatGPT", "system", "connectionist", None,
     "RLHF plus a chat box. 100 million users in two months. The interface is the invention."),
    (2022, "Chinchilla", "idea", "connectionist", "data",
     "We were training too big on too little. Data becomes the binding constraint."),
    (2023, "Llama and open weights", "system", "connectionist", None,
     "The open/proprietary split becomes a strategic question, not a licence detail."),
    (2023, "Mamba and state-space models", "idea", "connectionist", "architecture",
     "Linear time, constant memory. The lean turn gets an architecture."),
    (2024, "Nobel Prizes to Hopfield, Hinton, and to AlphaFold", "institution", "connectionist", None,
     "Physics and Chemistry. The two frontier currents, recognised in one week."),
    (2024, "EU AI Act enters into force", "institution", "neither", None,
     "Regulation becomes an engineering requirement. Week 2 begins here."),
    (2025, "DeepSeek moment", "shock", "connectionist", None,
     "Near-frontier quality at a fraction of assumed cost. 'Spend equals capability' cracks."),
    (2025, "Reasoning models", "idea", "connectionist", "compute",
     "Scale the thinking, not the parameters. Test-time compute as a new axis."),
]

WINTERS = [(1974, 1980, "AI Winter I"), (1987, 1993, "AI Winter II")]

ERAS = [
    (1943, 1956, "Prehistory", "The inference assembles: brains compute, so machines can."),
    (1956, 1969, "Golden years", "Everything works, in the shallow end."),
    (1969, 1980, "Winter I", "A correct theorem, read too broadly, plus exponential walls."),
    (1980, 1987, "Expert systems", "A smaller, sellable promise. Real money, real limits."),
    (1987, 1993, "Winter II", "The industry collapses; the field renames itself."),
    (1993, 2011, "Statistical turn", "AI colonises daily life under other names."),
    (2012, 2017, "Deep learning", "Depth, at last affordable."),
    (2017, 2023, "Scaling era", "Attention, and progress you can purchase."),
    (2023, 2027, "The two turns", "Past brute force: toward science, and toward lean."),
]


def load() -> pd.DataFrame:
    """The timeline as a tidy DataFrame."""
    df = pd.DataFrame(MILESTONES,
                      columns=["year", "name", "kind", "family", "ingredient", "note"])
    return df.sort_values("year").reset_index(drop=True)


# When did each ingredient become AVAILABLE AT COMMODITY SCALE? This is a
# different question from when it was invented, and the distinction is the
# whole point of the Week 1 lecture's most important slide. A convolutional
# architecture existed in 1979; a way to train one existed in 1986; but nobody
# could feed it until 2009 or afford to run it until 2007.
#
# These three dates are DEFENSIBLE, NOT DEFINITIVE. Argue with them: that is
# Exercise 2, and the argument is where the learning is.
SUPPLY = {
    "architecture": (1986, "Backpropagation popularised: depth becomes trainable in principle."),
    "compute": (2007, "CUDA: commodity gaming silicon becomes general-purpose parallel compute."),
    "data": (2009, "ImageNet: labelled data at a scale nobody had assembled before."),
}

# The first system to actually COMBINE all three. Note the gap: the supplies
# were all in place three years earlier. Availability is not assembly.
FIRST_COMBINATION = (2012, "AlexNet")


def ingredients_available(supply: dict | None = None) -> pd.DataFrame:
    """For each year, which of the three ingredients is available at scale?

    Returns a frame with boolean columns and `all_three`. Pass your own
    `supply` dict to test a different argument; the exercise is to defend one.
    """
    supply = SUPPLY if supply is None else supply
    rows = []
    for y in range(1940, 2027):
        have = {k: y >= v[0] for k, v in supply.items()}
        rows.append({"year": y, **have, "all_three": all(have.values())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    print(f"{len(df)} milestones, {df.year.min()} to {df.year.max()}\n")
    print(df.family.value_counts().to_string(), "\n")
    ing = ingredients_available()
    first = int(ing[ing.all_three].year.min())
    print("Ingredient supply dates (arguable, and that is Exercise 2):")
    for k, (yr, why) in SUPPLY.items():
        print(f"  {k:13s} {yr}   {why}")
    print(f"\nAll three available from: {first}")
    print(f"First system to combine them: {FIRST_COMBINATION[1]}, {FIRST_COMBINATION[0]}")
    print(f"\nThe gap is {FIRST_COMBINATION[0] - first} years. Availability is not assembly.")
