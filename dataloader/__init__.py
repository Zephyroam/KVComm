from .countries import CountriesEvaluator
from .tipsheets import TipsheetsEvaluator
from .hotpotqa import HotpotQAEvaluator
from .qasper import QaSperEvaluator
from .musique import MuSiQueEvaluator
from .multifieldqa_en import MultiFieldQAEnEvaluator
from .twowikimqa import TwoWikiMQAEvaluator
from .tmath import TMathEvaluator
from .repobench import RepoBenchEvaluator
from .samsum import SAMSumEvaluator

def get_evaluator(test_task: str):
    if test_task == "countries":
        return CountriesEvaluator()
    elif test_task == "tipsheets":
        return TipsheetsEvaluator()
    elif test_task == "hotpotqa":
        return HotpotQAEvaluator()
    elif test_task == "hotpotqa_full":
        return HotpotQAEvaluator(n_samples=None)
    elif test_task == "qasper":
        return QaSperEvaluator()
    elif test_task == "qasper_full":
        return QaSperEvaluator(n_samples=None)
    elif test_task == "musique":
        return MuSiQueEvaluator()
    elif test_task == "musique_full":
        return MuSiQueEvaluator(n_samples=None)
    elif test_task == "multifieldqa_en":
        return MultiFieldQAEnEvaluator()
    elif test_task == "twowikimqa":
        return TwoWikiMQAEvaluator()
    elif test_task == "tmath":
        return TMathEvaluator()
    elif test_task == "repobench":
        return RepoBenchEvaluator()
    elif test_task == "samsum":
        return SAMSumEvaluator()
    else:
        raise ValueError(f"Unsupported task name: {test_task}")

def get_multi_agent_evaluator(test_task: str):
    if test_task == "hotpotqa":
        return HotpotQAEvaluator(multi_agent=True)
    elif test_task == "musique":
        return MuSiQueEvaluator(multi_agent=True)
    elif test_task == "twowikimqa":
        return TwoWikiMQAEvaluator(multi_agent=True)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")

def get_mix_evaluator(test_task: str, mix_method: str):
    if test_task == "countries_tipsheets":
        from .countries_tipsheets import CountriesTipsheetsEvaluator
        return CountriesTipsheetsEvaluator(mix_method=mix_method)
    elif test_task == "countries_multifieldqa":
        from .countries_multifieldqa_en import CountriesMultiFieldQAEvaluator
        return CountriesMultiFieldQAEvaluator(mix_method=mix_method)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")