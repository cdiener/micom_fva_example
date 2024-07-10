from cobra.flux_analysis import flux_variability_analysis
from cobra.util import OptimizationError
from micom import load_pickle
from micom.logger import logger
from micom.media import minimal_medium
from micom.util import _apply_min_growth
from micom.workflows import workflow
from micom.workflows.media import process_medium
import pandas as pd
from os import path

def fva_worker(args):
    """Perform FVA for a set of taxa in a single model.

    Parameters
    ----------
    args : tuple of (micom.Community, list, float, medium, float)
        Specifies the model, taxa of interest, tradeoff, medium, and fva_threshold for the simulation.

    Returns
    -------
    pandas.DataFrame
        The results of the FVA.
    """
    path, mets, taxa, tradeoff, medium, thresh, strategy = args
    mets = set(mets)
    taxa = set(taxa)
    com = load_pickle(path)
    com.solver.configuration.presolve = False

    # Get exchange reactions for the taxa of interest only
    rxns = [r for r in com.internal_exchanges if r.community_id in taxa]
    if mets is not None:
        rxns = [r for r in rxns if list(r.reactants)[0].global_id in mets]
    if len(rxns) == 0:
        logger.info("Sample %s did not contain any of the requested taxa or metabolites." % com.id)
        return None
    logger.info("Found %d reactions to run FVA on." % len(rxns))

    ex_ids = [r.id for r in com.exchanges]
    logger.info(
        "%d/%d import reactions found in model.",
        medium.index.isin(ex_ids).sum(),
        len(medium),
    )
    com.medium = medium[medium.index.isin(ex_ids)]

    # Get growth rates
    try:
        sol = com.cooperative_tradeoff(fraction=tradeoff, fluxes=False, pfba=False)
    except Exception:
        logger.error(
            "Could not solve cooperative tradeoff for %s. "
            "This can often be fixed by enabling `presolve`, choosing more "
            "permissive atol and rtol arguments, or by checking that medium "
            "fluxes are > atol." % com.id
        )
        return None


    if strategy == "minimal imports":
        # Get the minimal medium and the solution at the same time
        res = minimal_medium(
            com,
            exchanges=None,
            community_growth=sol.growth_rate,
            min_growth=sol.members.growth_rate.drop("medium"),
            atol = 1e-12, # to get more low flux media elements
            solution=True
        )
        com.medium = res["medium"]
        sol = res["solution"]

    rates = sol.members.growth_rate.drop("medium")

    # Limit the individual growth rates to threshold %
    _apply_min_growth(com, thresh * rates)
    # The previous min growth constraints also constrain the community biomass so no constraint needed
    try:
        fva = flux_variability_analysis(
            model=com,
            reaction_list=rxns,
            fraction_of_optimum=0.0,
            processes=1
        )
    except Exception:
        logger.error(
            "Could not solve FVA for %s. "
            "You may try with strategy='none' or different threshold." % com.id
        )
        return None
    meta = fva.index.str.split("__", expand=True).to_frame()
    meta.index = fva.index
    meta.columns = ["reaction", "taxon"]
    fva = pd.concat([meta, fva], axis=1)
    fva["sample_id"] = com.id
    return fva


def eFVA(
    manifest,
    model_folder,
    medium,
    metabolites,
    taxa,
    tradeoff,
    strategy="minimal imports",
    fva_threshold=0.95,
    threads=1
):
    """Perform an exchange FVA for a list of models.

    This will perform flux variability analysis for a set of metabolite exchnages in
    one or several taxa. Note that the run time will scale with n_mets * n_taxa so running
    the analysis for 3 taxa and 3 metabolites will take 9 times as long as 1 metabolite in one
    taxon.

    Note
    ----

    To include net export and uptakes in the community add "medium" to the `taxa` argument.

    Parameters
    ----------
    manifest : pandas.DataFrame
        The manifest as returned by the `build` workflow.
    model_folder : str
        The folder in which to find the files mentioned in the manifest.
    medium : pandas.DataFrame
        A growth medium. Must have columns "reaction" and "flux" denoting
        exchange reactions and their respective maximum flux.
    metabolites : list of str
        IDs of metabolites for which we will look for exchange reactions in the taxa.
    taxa : list of str
        Names of taxa for which to perform the FVA.
    tradeoff : float in (0.0, 1.0]
        A tradeoff value. Can be chosen by running the `tradeoff` workflow or
        by experince. Tradeoff values of 0.5 for metagenomcis data and 0.3 for
        16S data seem to work well.
    strategy : str in ["none", "minimal imports"]
        The strategy for the medium.
    fva_threshold : float in (0.0, 1.0]
        The percentage of the maximum growth rate all taxa have to achieve after running cooperative tradeoff.
        A value of 0.95 means that each taxon has to achieve at least 95% of its maximum ctFBA gwrowth rate.
    threads : int >=1
        The number of parallel workers to use when building models. As a
        rule of thumb you will need around 1GB of RAM for each thread.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing
    """
    samples = manifest.sample_id.unique()
    paths = {
        s: path.join(model_folder, manifest[manifest.sample_id == s].file.iloc[0])
        for s in samples
    }
    medium = process_medium(medium, samples)
    args = [
        [
            p,
            metabolites,
            taxa,
            tradeoff,
            medium.flux[medium.sample_id == s],
            fva_threshold,
            strategy
        ]
        for s, p in paths.items()
    ]
    results = workflow(fva_worker, args, threads, progress=True)
    if all([r is None for r in results]):
        raise OptimizationError(
            "All numerical optimizations failed. This indicates a problem "
            "with the solver or numerical instabilities. Check that you have "
            "CPLEX or Gurobi installed. You may also increase the abundance "
            "cutoff to create simpler models."
        )
    results = pd.concat(r for r in results if r is not None).reset_index(drop=True)

    return results