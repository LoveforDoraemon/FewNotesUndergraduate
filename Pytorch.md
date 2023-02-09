# Pytorch

[TOC]

## MPNN

### parse

- parse_pdb

```python
def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)} # {A:0,R:1,...}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)} # {0:A,1:R,...}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip() # delete ' '

    if line[:6] == "HETATM" and line[17:17+3] == "MSE": # 非标准残基的原子. 记述非标准残基(标准氨基酸以及核酸以外的化合物, 包括抑制剂, 辅因子, 离子, 溶剂)中各原子的原子名称, 残基名称, 直角坐标(单位埃), 占有率, 温度因子等信息. 与ATOM记录的唯一区别在于HETATM残基默认情况下不会与其他残基相连. 注意, 水分子也应放在此记录中
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip() # 残基编号
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
            
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'
```

### loss

```python
def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av
```

### seq recovery

```python
seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])


# torch.nn.functional.one_hot(tensor,num_classes=)
x = torch.tensor([1, 1, 1, 3, 3, 4, 8, 5])
y1 = F.one_hot(x) # 只有一个参数张量x

x = tensor([1, 1, 1, 3, 3, 4, 8, 5])
x_shape = torch.Size([8])
y = tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0]])
y_shape = torch.Size([8, 9])


```

### 

```python
def tied_featurize(batch, device, chain_dict, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None, bias_by_res_dict=None, ca_only=False):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0*np.ones([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    #shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[b['name']] #masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10]=='seq_chain_']
            visible_chains = []
        num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains
        #random.shuffle(all_chains)
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']) #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:,None,:]
                else:
                    x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]               
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

       
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict!=None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx+v[0][v_count]-1)#make 0 to be the first
                                tied_beta[start_idx+v[0][v_count]-1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx+v_-1)#make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)


 
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list,0) #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list, 0)  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))

        pssm_coef_all[i,:] = pssm_coef_pad
        pssm_bias_all[i,:] = pssm_bias_pad
        pssm_log_odds_all[i,:] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[i,:] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)


    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:,:,0]
    else:
        X_out = X
    return X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta
```



### question

1. ```python
   line = line.replace("MSE","MET")
   
   if resn[-1].isalpha(): 
   ```

2. 