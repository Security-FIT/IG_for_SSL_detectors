#!/usr/bin/env python3
import sys
import argparse
import random
import os
import numpy as np
from collections import defaultdict

# Fixed random seed for reproducibility
random.seed(42)

def load_protocol(protocol_file):
    """
    Load protocol info: filename -> {speaker, attack, label}
    """
    info = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) >= 9:
                fid = parts[1]
                spk = parts[0]
                gender = parts[2]
                codec = parts[3]
                label = parts[8]
                attack = parts[7] if label == 'spoof' else 'bonafide'
                
                info[fid] = {
                    'spk': spk,
                    'gender': gender,
                    'codec': codec,
                    'attack': attack,
                    'label': label
                }
    return info

def load_list(list_path):
    with open(list_path, 'r') as f:
        return set([line.strip() for line in f if line.strip()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--subsets_dir", default="outputs/subsets")
    parser.add_argument("--output", default="outputs/final_selection_100.txt")
    
    # We now need all three score files to determine consensus correctness in the middle ground
    parser.add_argument("--aasist", required=True, help="Path to AASIST scores")
    parser.add_argument("--camhfa", required=True, help="Path to CAM++ scores")
    parser.add_argument("--sls", required=True, help="Path to SLS scores")
    
    args = parser.parse_args()

    # 1. Load Protocol
    print("Loading protocol...")
    proto = load_protocol(args.protocol)

    # 2. Load Subsets
    print("Loading consensus subsets...")
    cr_set = load_list(os.path.join(args.subsets_dir, "consensus_confident_right.txt"))
    cw_set = load_list(os.path.join(args.subsets_dir, "consensus_confident_wrong.txt"))
    mid_set = load_list(os.path.join(args.subsets_dir, "consensus_middle_ground.txt"))

    # 3. Load Scores for Middle Ground Logic (Consensus Correct/Wrong)
    print("Loading scores to split Middle Ground (checking consensus correctness)...")
    
    def load_scores_for_mid(score_file):
        """Loads scores and returns dict {fid: score}, plus labels/scores lists for EER calc"""
        sc_dict = {}
        lbl_list = []
        sc_list = []
        with open(score_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    fid = parts[0]
                    sc = float(parts[1])
                    lbl = int(parts[2])
                    sc_dict[fid] = sc
                    lbl_list.append(lbl)
                    sc_list.append(sc)
        return sc_dict, lbl_list, sc_list

    # Load all 3
    print("  Loading AASIST...")
    sc_aasist, l_aasist, s_aasist = load_scores_for_mid(args.aasist)
    print("  Loading CAMHFA...")
    sc_camhfa, l_camhfa, s_camhfa = load_scores_for_mid(args.camhfa)
    print("  Loading SLS...")
    sc_sls, l_sls, s_sls = load_scores_for_mid(args.sls)
    
    # Calculate EER Thresholds for each
    from sklearn.metrics import det_curve
    def get_eer_thresh(labels, scores):
        fpr, fnr, thresholds = det_curve(labels, scores, pos_label=1)
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        return thresholds[idx]

    thresh_aasist = get_eer_thresh(l_aasist, s_aasist)
    thresh_camhfa = get_eer_thresh(l_camhfa, s_camhfa)
    thresh_sls = get_eer_thresh(l_sls, s_sls)
    
    print(f"  EER Thresholds: AASIST={thresh_aasist:.4f}, CAMHFA={thresh_camhfa:.4f}, SLS={thresh_sls:.4f}")

    mid_correct = []
    mid_wrong = []
    
    # Check consensus on Middle Ground files
    for fid in mid_set:
        # We need the ground truth (label is in protocol or score file)
        # Using protocol info
        if fid not in proto: continue
        label_str = proto[fid]['label']
        label_int = 1 if label_str == 'bonafide' else 0
        
        # Get scores
        if fid not in sc_aasist or fid not in sc_camhfa or fid not in sc_sls:
            continue
            
        s1 = sc_aasist[fid]
        s2 = sc_camhfa[fid]
        s3 = sc_sls[fid]
        
        # Predictions (Score > EER => 1)
        p1 = 1 if s1 > thresh_aasist else 0
        p2 = 1 if s2 > thresh_camhfa else 0
        p3 = 1 if s3 > thresh_sls else 0
        
        # Consensus Correct: All 3 match the label
        if p1 == label_int and p2 == label_int and p3 == label_int:
            mid_correct.append(fid)
        # Consensus Wrong: All 3 mismatch the label
        elif p1 != label_int and p2 != label_int and p3 != label_int:
            mid_wrong.append(fid)
            
    print(f"  Consensus Middle Correct: {len(mid_correct)}")
    print(f"  Consensus Middle Wrong:   {len(mid_wrong)}")

    final_selection = []
    debug_log = []

    def log(msg):
        print(msg)
        debug_log.append(msg)

    # --- SELECTION 1: 32 Confident Right SPOOF (2 per attack) ---
    log("\n1. Selecting 32 Confident Right SPOOFs (2 per attack)...")
    cr_spoofs = [f for f in cr_set if proto[f]['label'] == 'spoof']
    
    # Group by attack
    by_attack = defaultdict(list)
    for fid in cr_spoofs:
        att = proto[fid]['attack']
        by_attack[att].append(fid)
    
    selected_cr_spoof = []
    attacks_found = sorted(by_attack.keys())
    log(f"  Attacks found in CR set: {attacks_found}")
    
    for att in attacks_found:
        candidates = by_attack[att]
        # Pick 2
        picked = random.sample(candidates, min(len(candidates), 2))
        selected_cr_spoof.extend(picked)
        
    log(f"  Selected {len(selected_cr_spoof)} CR Spoofs.")
    final_selection.extend(selected_cr_spoof)

    # --- SELECTION 2: 28 Confident Right BONAFIDE (Different Speakers) ---
    log("\n2. Selecting 28 Confident Right BONAFIDEs (Diff Speakers)...")
    cr_bonafide = [f for f in cr_set if proto[f]['label'] == 'bonafide']
    
    # Group by speaker
    by_spk = defaultdict(list)
    for fid in cr_bonafide:
        spk = proto[fid]['spk']
        by_spk[spk].append(fid)
        
    unique_spks = list(by_spk.keys())
    random.shuffle(unique_spks)
    
    selected_cr_bonafide = []
    # Try to pick 1 per speaker until we reach 28
    # Or just pick 28 speakers and 1 file from each
    target_bonus = 28
    
    if len(unique_spks) >= target_bonus:
        chosen_spks = unique_spks[:target_bonus]
        for spk in chosen_spks:
            selected_cr_bonafide.append(random.choice(by_spk[spk]))
    else:
        # Not enough speakers? (Usually abundant)
        # Take 1 from each, then fill up
        for spk in unique_spks:
            selected_cr_bonafide.append(random.choice(by_spk[spk]))
        
        needed = target_bonus - len(selected_cr_bonafide)
        remainders = [f for f in cr_bonafide if f not in selected_cr_bonafide]
        selected_cr_bonafide.extend(random.sample(remainders, min(len(remainders), needed)))
        
    log(f"  Selected {len(selected_cr_bonafide)} CR Bonafides.")
    final_selection.extend(selected_cr_bonafide)

    # --- SELECTION 3: 10 Confident Wrong BONAFIDE ---
    log("\n3. Selecting 10 Confident Wrong BONAFIDE...")
    cw_bonafide = [f for f in cw_set if proto[f]['label'] == 'bonafide']
    selected_cw_bonafide = random.sample(cw_bonafide, min(len(cw_bonafide), 10))
    log(f"  Selected {len(selected_cw_bonafide)} CW Bonafides (Available: {len(cw_bonafide)})")
    final_selection.extend(selected_cw_bonafide)

    # --- SELECTION 4: 10 Confident Wrong SPOOF ---
    log("\n4. Selecting 10 Confident Wrong SPOOF...")
    cw_spoof = [f for f in cw_set if proto[f]['label'] == 'spoof']
    
    # Try to diversify attacks
    cw_spoof_by_attack = defaultdict(list)
    for f in cw_spoof:
        cw_spoof_by_attack[proto[f]['attack']].append(f)
        
    cw_spoof_attacks = list(cw_spoof_by_attack.keys())
    log(f"  CW Spoof Attacks Available: {cw_spoof_attacks}")
    
    selected_cw_spoof = []
    # Round robin selection from attacks
    while len(selected_cw_spoof) < 10 and cw_spoof_attacks:
        for att in cw_spoof_attacks[:]: # iterate copy
            if not cw_spoof_by_attack[att]:
                cw_spoof_attacks.remove(att)
                continue
                
            choice = random.choice(cw_spoof_by_attack[att])
            selected_cw_spoof.append(choice)
            cw_spoof_by_attack[att].remove(choice)
            
            if len(selected_cw_spoof) >= 10:
                break
                
    log(f"  Selected {len(selected_cw_spoof)} CW Spoofs (Available: {len(cw_spoof)})")
    final_selection.extend(selected_cw_spoof)

    # --- SELECTION 5: 5 Spoof / 5 Bonafide Correct Middle ---
    log("\n5. Selecting 5 Spoof / 5 Bonafide Correct Middle...")
    mid_corr_spoof = [f for f in mid_correct if proto[f]['label'] == 'spoof']
    mid_corr_bona = [f for f in mid_correct if proto[f]['label'] == 'bonafide']
    
    # Spread across attacks for spoof
    mid_corr_spoof_sel = []
    # Simple strategy: unique attacks random sample
    temp_attacks_map = defaultdict(list)
    for f in mid_corr_spoof: temp_attacks_map[proto[f]['attack']].append(f)
    att_keys = list(temp_attacks_map.keys())
    random.shuffle(att_keys)
    
    # Pick 1 from 5 diff attacks
    for i in range(5):
        if i < len(att_keys):
            att = att_keys[i]
            mid_corr_spoof_sel.append(random.choice(temp_attacks_map[att]))
        elif mid_corr_spoof: # fallback
            mid_corr_spoof_sel.append(random.choice(mid_corr_spoof))
            
    mid_corr_bona_sel = random.sample(mid_corr_bona, min(len(mid_corr_bona), 5))
    
    log(f"  Selected {len(mid_corr_spoof_sel)} Mid-Correct Spoofs")
    log(f"  Selected {len(mid_corr_bona_sel)} Mid-Correct Bonafides")
    
    final_selection.extend(mid_corr_spoof_sel)
    final_selection.extend(mid_corr_bona_sel)

    # --- SELECTION 6: 5 Spoof / 5 Bonafide Wrong Middle ---
    log("\n6. Selecting 5 Spoof / 5 Bonafide Wrong Middle...")
    mid_wrong_spoof = [f for f in mid_wrong if proto[f]['label'] == 'spoof']
    mid_wrong_bona = [f for f in mid_wrong if proto[f]['label'] == 'bonafide']
    
    mid_wrong_spoof_sel = random.sample(mid_wrong_spoof, min(len(mid_wrong_spoof), 5))
    mid_wrong_bona_sel = random.sample(mid_wrong_bona, min(len(mid_wrong_bona), 5))
    
    log(f"  Selected {len(mid_wrong_spoof_sel)} Mid-Wrong Spoofs")
    log(f"  Selected {len(mid_wrong_bona_sel)} Mid-Wrong Bonafides")
    
    final_selection.extend(mid_wrong_spoof_sel)
    final_selection.extend(mid_wrong_bona_sel)

    # --- SELECTION 7: 2 Worst-Scored Spoofs per Attack (Highest Aasist Score) ---
    log("\n7. Selecting 2 'Worst-Scored' Spoofs per Attack (Highest Scores)...")
    
    spoogs_by_attack_all = defaultdict(list)
    for fid, info in proto.items():
        if info['label'] == 'spoof':
            if fid in sc_aasist:
                spoogs_by_attack_all[info['attack']].append((fid, sc_aasist[fid]))
    
    worst_per_attack = []
    attacks_all = sorted(spoogs_by_attack_all.keys())
    
    for att in attacks_all:
        candidates = spoogs_by_attack_all[att]
        # Sort descending by score (Higher score = More Bonafide-like = Worse detection for spoof)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 2
        top2 = [x[0] for x in candidates[:2]]
        worst_per_attack.extend(top2)
        
    log(f"  Selected {len(worst_per_attack)} Worst Spoofs (2 from each of {len(attacks_all)} attacks).")
    
    final_selection.extend(worst_per_attack) # This might introduce duplicates in the list
    
    selection_reasons = defaultdict(list)
    
    def register_reasons(file_list, reason):
        for fid in file_list:
            selection_reasons[fid].append(reason)
            
    register_reasons(selected_cr_spoof, "CR Spoof")
    register_reasons(selected_cr_bonafide, "CR Bonafide")
    register_reasons(selected_cw_bonafide, "CW Bonafide")
    register_reasons(selected_cw_spoof, "CW Spoof")
    register_reasons(mid_corr_spoof_sel, "Mid-Correct Spoof")
    register_reasons(mid_corr_bona_sel, "Mid-Correct Bonafide")
    register_reasons(mid_wrong_spoof_sel, "Mid-Wrong Spoof")
    register_reasons(mid_wrong_bona_sel, "Mid-Wrong Bonafide")
    register_reasons(worst_per_attack, "Worst-Scored Spoof")
    
    # Final consolidated list
    
    ordered_final = []
    seen_ids = set()
    for fid in final_selection:
        if fid not in seen_ids:
            ordered_final.append(fid)
            seen_ids.add(fid)
            
    print(f"\nTotal unique selected: {len(ordered_final)}")
    
    with open(args.output, 'w') as f:
        for fid in ordered_final:
            f.write(f"{fid}\n")

    # Also save a detailed CSV
    detail_path = args.output.replace(".txt", "_details.csv")
    with open(detail_path, 'w') as f:
        f.write("FileID,Label,Attack,Speaker,Gender,Codec,SelectionReason,AASIST_Score,CAMHFA_Score,SLS_Score\n")
        for fid in ordered_final:
            info = proto.get(fid, {})
            lbl = info.get('label', '?')
            att = info.get('attack', '?')
            spk = info.get('spk', '?')
            gen = info.get('gender', '?')
            cod = info.get('codec', '?')
            reasons = "+".join(selection_reasons[fid])
            s1 = sc_aasist.get(fid, '?')
            s2 = sc_camhfa.get(fid, '?')
            s3 = sc_sls.get(fid, '?')
            f.write(f"{fid},{lbl},{att},{spk},{gen},{cod},{reasons},{s1},{s2},{s3}\n")
            
    print(f"Saved list to {args.output}")

if __name__ == "__main__":
    main()
