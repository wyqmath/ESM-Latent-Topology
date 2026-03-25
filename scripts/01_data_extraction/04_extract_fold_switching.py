#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Extract fold-switching protein sequences from paper supplementary material
Combines paper extraction and manual sequence addition
Input: data/raw/pnas.1800168115.sapp_yellow.md
Output: data/fold_switching_paper_full_length.fasta, data/fold_switching_paper_regions.json
"""
import re
import os
import json

# ============ Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_FILE = os.path.join(BASE_DIR, 'data/raw/pnas.1800168115.sapp_yellow.md')
IFU_FASTA = os.path.join(BASE_DIR, 'data/fold_switching_ifu_regions.fasta')
OUTPUT_FASTA = os.path.join(BASE_DIR, 'data/fold_switching_paper_full_length.fasta')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data/fold_switching_paper_regions.json')

# Manual sequences for 8 missing proteins
MANUAL_SEQUENCES = {
    '5lj3M': 'MTSWRDKSAKVQVKESELPSSIPAQTGLTFNIWYNKWSQGFAGNTRFVSPFALQPQLHSGKTRGDNDGQLFFCLFFAKGMCCLGPKCEYLHHIPDEEDIGKLALRTEALDCFGREKFADYREDMGGIGSFRKKNKTLYVGGIDGALNSKHLKPAQIESRIRFVFSRLGDIDRIRYVESKNCGFVKFKYQANAEFAKEAMSNQTLLLPSDKEWDDRREGTGLLVKWANEDPDPAAQKRLQEELKLESLNMMVHLINNNTNSAGTEVNNKNNERLDRTFPEASVDNVKKRLLPLDNGMESDDFIEKLKKVKKNISRENISSKPSVGKLGGPLLDYLSSDED',

    '4rr2D': 'MEFSGRKWRKLRLAGDQRNASYPHCLQFYLQPPSENISLIEFENLAIDRVKLLKSVENLGVSYVKGTEQYQSKLESELRKLKFSYRENLEDEYEPRRRDHISHFILRLAYCQSEELRRWFIQQEMDLLRFRFSILPKDKIQDFLKDSQLQFEAISDEEKTLREQEIVASSPSLSGLKLGFESIYKIPFADALDLFRGRKVYLEDGFAYVPLKDIVAIILNEFRAKLSKALALTARSLPAVQSDERLQPLLNHLSHSYTGQDYSTQGNVGKISLDQIDLLSTKSFPPCMRQLHKALRENHHLRHGGRMQYGLFLKGIGLTLEQALQFWKQEFIKGKMDPDKFDKGYSYNIRHSFGKEGKRTDYTPFSCLKIILSNPPSQGDYHGCPFRHSDPELLKQKLQSYKISPGGISQILDLVKGTHYQVACQKYFEMIHNVDDCGFSLNHPNQFFCESQRILNGGKDIKKEPIQPETPQPKPSVQKTKDASSALASLNSSLEMDMEGLEDYFSEDS',

    '2ougC': 'MQSWYLLYCKRGQLQRAQEHLERQAVNCLAPMITLEKIVRGKRTAVSEPLFPNYLFVEFDPEVIHTTTINATRGVSHFVRFGASPAIVPSAVIHQLSVYKPKDIVDPATPYPGDKVIITEGAFEGFQAIFTEPDGEARSMLLLNLINKEIKHSVKNTEFRKL',

    '5ly6B': 'MANKAVNDFILAMNYDKKKLLTHQGESIENRFIKEGNQLPDEFVVIERKKRSLSTNTSDISVTATNDSRLYPGALLVVDETLLENNPTLLAVDRAPMTYSIDLPGLASSDSFLQVEDPSNSSVRGAVNDLLAKWHQDYGQVNNVPARMQYEKITAHSMEQLKVKFGSDFEKAANSLDIDFNAVHSGEKQIQIVNFKQIYYTVSVDAVKNPGDVFQDTVTVEDLKQRGISAERPLVYISSVAYGRQVYLKLETTSKSDEVQAAFEAAILGVKVAPQTQWKQILDNTEVKAVILGGDPSSGARVVTGKVDMVEDLIQEGSRFTADHPGLPISYTTSFLRDNVVATFQNSTDYVETKVTAYRNGDLLLDHSGAYVAQYYITWDELSYDHQGKEVLTPKAWDRNGQDLTAHFTTSIPLKGNVRNLSVKIRECTGLAWEWWRTVYEKTDLPLVRKRTISIWGTTLYPQVEDKVEND',

    '4rmbA': 'RIMKLDDERQTETYITQINPEGKEMYFASGLGNLYTIIGSDGTSGSPVNLLNAEVKILKTNSKNLTDSMDQNYDSPEFEDVTSQYSYTNDGSKITIDWKTNSISSTTSYVVLVKIPKQSGVLYSTVSDINQTYGSKYSYGHTNISGDSDANAEIKLLSLEHHHHHH',

    '3j9cA': 'TVPDRDNDGIPDSLEVEGYTVDVKNKRTFLSPWISNIHEKKGLTKYKSSPEKWSTASDPYSDFEKVTGRIDKNVSPEARHPLVAAYPIVHVDMENIILSKNEDQSTQNTDSQTRTISKNTSTSRTHTSEVHGNAEVHASFFDIGGSVSAGFSNSNSSTVAIDHSLSLAGERTWAETMGLNTADTARLNANIRYVNTGTAPIYNVLPTTSLVLGKNQTLATIKAKENQLSQILAPNNYYPSKNLAPIALNAQDDFSSTPITMNYNQFLELEKTKQLRLDTDQVYGNIATYNFENGRVRVDTGSNWSEVLPQIQETTARIIFNGKDLNLVERRIAAVNPSDPLETTKPDMTLKEALKIAFGFNEPNGNLQYQGKDITEFDFNFDQQTSQNIKNQLAELNATNIYTVLDKIKLNAKMNILIRDKRFHYDRNNIAVGADESVVKEAHREVINSSTEGLLLNIDKDIRKILSGYIVEIEDTEGLKEVINDRYDMLNISSLRQDGKTFIDFKKYNDKLPLYISNPNYKVNVYAVTKENTIINPSENGDTSTNGIKKILIFSKKGYEIG',

    '4cmqB': 'GAASMDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD',

    '3o44A': 'SGFASPAPANSETNTLPHVAFYISVNRAISDEECTFNNSWLWKNEKGSRPFCKDANISLIYRVNLERSLQYGIVGSATPDAKIVRISLDDDSTGAGIHLNDQLGYRQFGASYTTLDAYFREWSTDAIAQDYRFVFNASNNKAQILKTFPVDNINEKFERKEVSGFELGVTGGVEVSGDGPKAKLEARASYTQSRWLTYNTQDYRIERNAKNAQAVSFTWNRQQYATAESLLNRSTDALWVNTYPVDVNRISPLSYASFVPKMDVIYKASATETGSTDFIIDSSVNIRPIYNGAYKHYYVVGAHQSYHGFEDTPRRRITKSASFTVDWDHPVFTGGRPVNLQLASFNNRCIQVDAQGRLTANMCDSQQSAQSFIYDQLGRYVSASNTKLCLDGAALDALQPCNQNLTQRWEWRKGTDELTNVYSGESLGHDKQTGELGLYASSNDAVSLRTITAYTDVFNAQESSPILGYTQGKMNQQRVGQDNRLYVRAGAAIDALGSASDLLVGGNGGSLSSVDLSGVKSITATSGDFQYGGQQLVALTFTYQDGRQQTVGSKAYVTNAHEDRFDLPDAAKITQLKIWADDWLVKGVQFDLN'
}


# ============ Utility Functions ============

def write_fasta(records, output_file):
    """Write sequences to FASTA file"""
    with open(output_file, 'w') as f:
        for seq_id, sequence in records:
            f.write(f">{seq_id}\n")
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')


def write_json(data, output_file):
    """Write data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def find_fragment_position(full_sequence, fragment):
    """Find the position of a fragment in the full sequence (1-indexed)"""
    full_seq = full_sequence.upper().replace('-', '')
    frag_seq = fragment.upper().replace('-', '')

    # Try exact match
    pos = full_seq.find(frag_seq)
    if pos != -1:
        return pos + 1, pos + len(frag_seq)

    # Try without first M
    if frag_seq.startswith('M') and len(frag_seq) > 1:
        pos = full_seq.find(frag_seq[1:])
        if pos != -1:
            return pos + 1, pos + len(frag_seq) - 1

    # Try fuzzy matching (allow 30% mismatches)
    frag_len = len(frag_seq)
    best_match = None
    best_mismatches = float('inf')

    for i in range(len(full_seq) - frag_len + 1):
        window = full_seq[i:i+frag_len]
        mismatches = sum(1 for a, b in zip(window, frag_seq) if a != b)

        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_match = (i + 1, i + frag_len)

            if mismatches == 0:
                break

    # Accept match if similarity >= 70%
    if best_match and best_mismatches <= frag_len * 0.3:
        return best_match

    # Try fuzzy matching without first M
    if frag_seq.startswith('M') and len(frag_seq) > 1:
        frag_seq_no_m = frag_seq[1:]
        frag_len = len(frag_seq_no_m)
        best_match = None
        best_mismatches = float('inf')

        for i in range(len(full_seq) - frag_len + 1):
            window = full_seq[i:i+frag_len]
            mismatches = sum(1 for a, b in zip(window, frag_seq_no_m) if a != b)

            if mismatches < best_mismatches:
                best_mismatches = mismatches
                best_match = (i + 1, i + frag_len)

                if mismatches == 0:
                    break

        if best_match and best_mismatches <= frag_len * 0.3:
            return best_match

    return None


def extract_paper_sequences(paper_file):
    """Extract sequences from paper supplementary material"""
    with open(paper_file, 'r') as f:
        content = f.read()

    # Extract table rows
    rows = re.findall(r'<tr><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td></tr>', content)

    # Group by PDB ID, keep only first entry
    pdb_sequences = {}
    for pdb, seq, qr in rows:
        pdb = pdb.strip()
        seq = seq.strip()

        # Skip header
        if 'PDB1' in pdb or 'prediction' in seq:
            continue

        if len(pdb) >= 5 and pdb[:4].isalnum():
            seq_clean = seq.replace(' ', '').replace('X', '').upper()
            seq_clean = ''.join(c for c in seq_clean if c.isalpha())

            if len(seq_clean) > 0 and pdb not in pdb_sequences:
                pdb_sequences[pdb] = seq_clean

    return pdb_sequences


def load_ifu_fragments(ifu_fasta):
    """Read IFU fragments"""
    fragments = {}
    with open(ifu_fasta, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    fragments[current_id] = ''.join(current_seq)
                # Parse header: >fold_switching|3zxgB|IFU:19-156
                # Extract just the PDB chain ID (3zxgB)
                header = line[1:]
                parts = header.split('|')
                if len(parts) >= 2:
                    current_id = parts[1]  # Extract PDB chain ID
                else:
                    current_id = header
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            fragments[current_id] = ''.join(current_seq)

    return fragments


def main():
    print("=" * 70)
    print("Extract Fold-Switching Sequences from Paper")
    print("=" * 70)

    # Step 1: Extract paper sequences
    print("\nStep 1: Extracting paper sequences...")
    paper_seqs = extract_paper_sequences(PAPER_FILE)
    print(f"  Paper sequences: {len(paper_seqs)}")

    # Step 2: Read IFU fragments
    print("\nStep 2: Reading IFU fragments...")
    ifu_fragments = load_ifu_fragments(IFU_FASTA)
    print(f"  IFU fragments: {len(ifu_fragments)}")

    # Step 3: Match IFU fragments to paper sequences
    print("\nStep 3: Matching IFU fragments to paper sequences...")

    matched_records = []
    region_data = []

    success_count = 0
    fail_count = 0

    for fasta_id, ifu_seq in ifu_fragments.items():
        # Extract PDB ID (format: "1g2cF" or similar)
        pdb_chain = fasta_id

        # Try paper sequences first
        if pdb_chain in paper_seqs:
            paper_seq = paper_seqs[pdb_chain]

            # Find IFU fragment position (supports fuzzy matching)
            result = find_fragment_position(paper_seq, ifu_seq)

            if result:
                ifu_start, ifu_end = result
                print(f"  ✅ {pdb_chain}: IFU at {ifu_start}-{ifu_end} (full length: {len(paper_seq)} aa)")

                # Create FASTA record
                fasta_header = f"fold_switching|{pdb_chain}|IFU:{ifu_start}-{ifu_end}"
                matched_records.append((fasta_header, paper_seq))

                # Create JSON record
                region_entry = {
                    "id": pdb_chain,
                    "sequence": paper_seq,
                    "length": len(paper_seq),
                    "regions": [
                        {
                            "type": "IFU",
                            "start": ifu_start,
                            "end": ifu_end
                        }
                    ]
                }
                region_data.append(region_entry)

                success_count += 1
            else:
                print(f"  ❌ {pdb_chain}: IFU fragment mismatch (IFU: {len(ifu_seq)} aa, paper: {len(paper_seq)} aa)")
                fail_count += 1
        elif pdb_chain in MANUAL_SEQUENCES:
            # Use manual sequence
            full_seq = MANUAL_SEQUENCES[pdb_chain]

            # Find IFU fragment position
            pos = full_seq.find(ifu_seq)
            if pos != -1:
                ifu_start = pos + 1  # 1-indexed
                ifu_end = pos + len(ifu_seq)

                print(f"  ✅ {pdb_chain}: IFU at {ifu_start}-{ifu_end} (manual, full length: {len(full_seq)} aa)")

                # Create FASTA record
                fasta_header = f"fold_switching|{pdb_chain}|IFU:{ifu_start}-{ifu_end}"
                matched_records.append((fasta_header, full_seq))

                # Create JSON record
                region_entry = {
                    "id": pdb_chain,
                    "sequence": full_seq,
                    "length": len(full_seq),
                    "regions": [
                        {
                            "type": "IFU",
                            "start": ifu_start,
                            "end": ifu_end
                        }
                    ]
                }
                region_data.append(region_entry)

                success_count += 1
            else:
                print(f"  ❌ {pdb_chain}: IFU fragment not found in manual sequence")
                fail_count += 1
        else:
            print(f"  ❌ {pdb_chain}: Not found in paper or manual sequences")
            fail_count += 1

    print(f"\n  Successfully matched: {success_count}")
    print(f"  Match failed: {fail_count}")

    # Step 4: Write output files
    print("\nStep 4: Writing output files...")
    write_fasta(matched_records, OUTPUT_FASTA)
    print(f"  FASTA written: {OUTPUT_FASTA}")

    output_data = {
        "dataset": "fold_switching",
        "source": "paper_supplementary_material_plus_manual",
        "total_sequences": len(region_data),
        "sequences": region_data
    }
    write_json(output_data, OUTPUT_JSON)
    print(f"  JSON written: {OUTPUT_JSON}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"  Total sequences: {len(matched_records)}")
    if matched_records:
        lengths = [len(seq) for _, seq in matched_records]
        print(f"  Length range: {min(lengths)} - {max(lengths)} aa")
        print(f"  Mean length: {sum(lengths) / len(lengths):.1f} aa")

    print(f"\n  Success rate: {success_count}/{len(ifu_fragments)} = {success_count/len(ifu_fragments)*100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
