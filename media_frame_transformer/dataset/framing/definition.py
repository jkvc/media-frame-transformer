# from os.path import join

# from config import DATA_DIR

# ISSUES = [
#     "climate",
#     "deathpenalty",
#     "guncontrol",
#     "immigration",
#     "samesex",
#     "tobacco",
# ]
# ISSUE2IIDX = {issue: i for i, issue in enumerate(ISSUES)}
# N_ISSUES = len(ISSUES)

# PRIMARY_FRAME_NAMES = [
#     "Economic",
#     "Capacity and Resources",
#     "Morality",
#     "Fairness and Equality",
#     "Legality, Constitutionality, Jurisdiction",
#     "Policy Prescription and Evaluation",
#     "Crime and Punishment",
#     "Security and Defense",
#     "Health and Safety",
#     "Quality of Life",
#     "Cultural Identity",
#     "Public Sentiment",
#     "Political",
#     "External Regulation and Reputation",
#     "Other",
# ]
# PRIMARY_FRAME_N_CLASSES = len(PRIMARY_FRAME_NAMES)


# def primary_frame_code_to_fidx(frame_float: float) -> int:
#     # see codes.json, non null frames are [1.?, 15.?], map them to [0, 14]
#     assert frame_float != 0
#     assert frame_float < 16
#     return int(frame_float) - 1


# LABELPROPS_DIR = join(DATA_DIR, "framing_labeled", "labelprops")
