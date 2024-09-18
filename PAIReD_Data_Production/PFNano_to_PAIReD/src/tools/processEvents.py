"""
 * Function: processEvents
 * -----------------------
Process the events to produce PAIReD jets with all relevant quantities.

Parameters
----------
Events : event tree object
    Object containing all information of multiple simulated events.
ttbar : bool
    True if the data file processed is a ttbar Event.


Returns
-------
DataPAIReD : dict
    A dictionary with the content summarized in /notes/PAIReD_ROOT_file_content.md
"""

# Include dependencies
import awkward as ak
import numpy as np
import vector
from tools.branchnames import BranchNames
from tools.helpers import isoLeptonCut, deltaPhi, getJetClusterIndex
from tools.getMCInfo import getMCInfo



def processEvents(Events, physics_process=0, PAIReD_geometry="Ellipse"):

    # import the PAIReD jet geometry
    if PAIReD_geometry=="Ellipse":
        from tools.PAIReD_geometries.ellipse import isInPAIReD
    elif PAIReD_geometry=="AK4":
        from tools.PAIReD_geometries.ak4 import isInPAIReD
    elif PAIReD_geometry=="Dumbbell":
        from tools.PAIReD_geometries.dumbbell import isInPAIReD

    # check if events have two or more jets that survive the selection cut
    Jetcut = (((Events.Jet_pt * (1-Events.Jet_rawFactor)) > 20) & 
                (abs(Events.Jet_eta) < 2.5) & 
                (isoLeptonCut(Events)) &
                (Events.Jet_jetId > 4)
            )
    Eventcut = (ak.sum(Jetcut, axis=1) > 1)

    # cut all events that do not meet this criterion
    Events = Events[Eventcut]

    # check if there are events with two or more jets
    if sum(Eventcut)==0:
        print("   !!! No event in the batch had more than one jet surviving the selection cut !!!")
        return False

    # ATTENTION!!! defined arrays are copies of the data!!!

    # sort the particles according to their pt
    s = ak.argsort(Events.PFCands_pt, ascending=False, axis=1)
    # and only include particles not considered pileup
    s = s[Events.PFCands_puppiWeight[s] != 0]
    Part = {"phi": Events.PFCands_phi[s], "eta": Events.PFCands_eta[s],
            "pt": Events.PFCands_pt[s], "charge": Events.PFCands_charge[s],
            "pid": Events.PFCands_pdgId[s], "d0val": Events.PFCands_d0[s],
            "d0err": Events.PFCands_d0Err[s], "dzval": Events.PFCands_dz[s],
            "dzerr": Events.PFCands_dzErr[s], "mass": Events.PFCands_mass[s],
            "puppiweight": Events.PFCands_puppiWeight[s],
            "jetindex": getJetClusterIndex(Events)[s]}
    
    # sort the secondary vertices (SV) according to their dlenSig
    s = ak.argsort(Events.SV_dlenSig, ascending=False, axis=1)
    SV = {k: Events["SV_"+k][s] for k in BranchNames["sv"][:16]}
    
    # define 4-vector for jets in order to calculate the energy
    Jetcut = (((Events.Jet_pt * (1-Events.Jet_rawFactor)) > 20) & 
            (abs(Events.Jet_eta) < 2.5) & 
            (isoLeptonCut(Events)) &
            (Events.Jet_jetId > 4)
            )
    Jet4 = vector.zip({"eta": Events.Jet_eta[Jetcut], "phi": Events.Jet_phi[Jetcut],
            "pt": Events.Jet_pt[Jetcut], "mass": Events.Jet_mass[Jetcut]})
    # create jet object
    Jet = ak.zip({"phi": Events.Jet_phi[Jetcut], "eta": Events.Jet_eta[Jetcut],
            "pt": Events.Jet_pt[Jetcut], "energy": Jet4.E,
            "nparticles": Events.Jet_nConstituents[Jetcut],
            "rawfactor": Events.Jet_rawFactor[Jetcut],
            "index": ak.local_index(Events.Jet_pt, axis=1)[Jetcut]})

    # get all jet pair combinations
    Jet = ak.combinations(Jet, 2, axis=1, fields=["j1", "j2"],
            replacement=False)
    # get also the combinations for the genJetIdx
    Jet_genJetIdx = ak.combinations(Events.Jet_genJetIdx[Jetcut], 2, 
            axis=1, fields=["j1", "j2"], replacement=False)

    # get particles inside the PAIReD jet
    isInPart = isInPAIReD(Jet.j1.eta, Jet.j2.eta, Jet.j1.phi, Jet.j2.phi,
                Part["eta"], Part["phi"])

    # get the generated particles inside the PAIReD jet
    isInGenPart = isInPAIReD(Jet.j1.eta, Jet.j2.eta, Jet.j1.phi, Jet.j2.phi,
                Events.GenPart_eta, Events.GenPart_phi)

    # get the secondary vertices inside the PAIReD jet
    isInSV = isInPAIReD(Jet.j1.eta, Jet.j2.eta, Jet.j1.phi, Jet.j2.phi,
                SV["eta"], SV["phi"])

    # make part arrays broadcastable with isInPart
    # and drop all particles that are not in the PAIReD jet
    for name in Part.keys():
        Part[name] = ak.unflatten(Part[name], 1, axis=0)
        Part[name], isInPart = ak.broadcast_arrays(Part[name], isInPart)
        Part[name] = Part[name][isInPart]

    # make SV arrays broadcastable with isInSV
    # and drop all SVs that are not in the PAIReD jet
    for name in SV.keys():
        SV[name] = ak.unflatten(SV[name], 1, axis=0)
        SV[name], isInSV = ak.broadcast_arrays(SV[name], isInSV)
        SV[name] = SV[name][isInSV]

    # flags indicating if particles are clustered in one of seed jets
    Part["in_jet1"] = Part["jetindex"] == Jet.j1.index
    Part["in_jet2"] = Part["jetindex"] == Jet.j2.index

    # calculate remaining variables (px,py,pz,E) from 4-vector
    Part4 = vector.zip({"eta": Part["eta"], "phi": Part["phi"],
                "pt": Part["pt"], "mass": Part["mass"]})
    Part["px"] = Part4.px
    Part["py"] = Part4.py
    Part["pz"] = Part4.pz
    Part["energy"] = Part4.E

    # calculate deta and dphi with respect to the jets
    # (part_eta - jet1_eta) * (jet1_eta > 0 ? 1 : -1)
    Part["deta1"] = (Part["eta"] - Jet.j1.eta) * (-1)**(Jet.j1.eta < 0)
    # (part_eta - jet2_eta) * (jet2_eta > 0 ? 1 : -1)
    Part["deta2"] = (Part["eta"] - Jet.j2.eta) * (-1)**(Jet.j2.eta < 0)
    # dphi
    Part["dphi1"] = deltaPhi(Part["phi"], Jet.j1.phi)
    Part["dphi2"] = deltaPhi(Part["phi"], Jet.j2.phi)

    # calculate the deta and dphi for the secondary vertices
    SV["deta1"] = (SV["eta"] - Jet.j1.eta) * (-1)**(Jet.j1.eta < 0)
    SV["deta2"] = (SV["eta"] - Jet.j2.eta) * (-1)**(Jet.j2.eta < 0)
    SV["dphi1"] = deltaPhi(SV["phi"], Jet.j1.phi)
    SV["dphi2"] = deltaPhi(SV["phi"], Jet.j2.phi)

    # calculate dijets from dijet 4-vector
    Dijet4 = vector.zip({"px": ak.sum(Part4.px, axis=2),
                    "py": ak.sum(Part4.py, axis=2),
                    "pz": ak.sum(Part4.pz, axis=2),
                    "E":  ak.sum(Part4.E, axis=2)})
    
    # get indices of the particles in one of the AK4 jets
    s = ak.local_index(Part["pt"], axis=2)[(Part["in_jet1"]) & (Part["in_jet2"])]
    # get indices of the particles in no jet
    s = ak.concatenate([s, ak.local_index(Part["pt"], axis=2)[(~Part["in_jet1"]) & (~Part["in_jet2"])]], axis=2)
    # bring particles clustered to the seed jets to the front of the list
    for name in Part.keys():
        Part[name] = Part[name][s]
    
    # prepare jet and particle objects for the tree
    part = ak.zip(Part)
    sv = ak.zip(SV)
    jet1 = Jet.j1
    jet2 = Jet.j2
    dijet = ak.zip({"eta": Dijet4.eta, "phi": Dijet4.phi,
                    "pt": Dijet4.pt, "mass": Dijet4.mass,
                    "nparticles": ak.num(part, axis=2),
                    "index": ak.local_index(Jet.j1, axis=1)})

    # prepare other single value branches
    ones = ak.ones_like(Jet.j1.phi)
    event = Events.event * ones
    genweight = Events.genWeight * ones
    run = Events.run * ones
    Pileup_nPU = Events.Pileup_nPU * ones
    pv_n = Events.PV_npvs * ones
    pv_ngood = Events.PV_npvsGood * ones
    fgrfcc = Events.Rho_fixedGridRhoFastjetCentralCalo * ones
    fgrfccpu = Events.Rho_fixedGridRhoFastjetCentralChargedPileUp * ones
    MC_physics_process = physics_process * ones

    # get summarized information from the MC simulation
    MCInfo = getMCInfo(Events, isInGenPart, Jet_genJetIdx, Jetcut)

    if MCInfo == False:
        return False

    DataPAIReD = {
        "event" : event,
        "genweight": genweight,
        "run" : run,
        "Pileup_nPU" : Pileup_nPU,
        "pv_n" : pv_n,
        "pv_ngood" : pv_ngood,
        "fgrfcc" : fgrfcc,
        "fgrfccpu" : fgrfccpu,
        "jet1" : jet1,
        "jet2" : jet2,
        "dijet" : dijet,
        "part" : part,
        "sv" : sv,
        "MC_physics_process" : MC_physics_process
    }

    for name in MCInfo.keys():
        DataPAIReD[name] = MCInfo[name]

    # remove the dimension of N_events
    for name in DataPAIReD.keys():
        DataPAIReD[name] = ak.flatten(DataPAIReD[name], axis=1)
    
    return DataPAIReD