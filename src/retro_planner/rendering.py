from rdkit.Chem import AllChem, Draw

from retro_planner.chemistry import mol_from_smiles_without_atom_maps


def generate_molecule_image(smiles: str):
    try:
        mol = mol_from_smiles_without_atom_maps(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(360, 260))
    except Exception as exc:
        print(f"Mol Img Error: {exc}")
        return None


def generate_reaction_image(reactants_smiles, product_smiles):
    try:
        reactants = [
            mol_from_smiles_without_atom_maps(reactant)
            for reactant in reactants_smiles
            if reactant
        ]
        product_values = (
            product_smiles
            if isinstance(product_smiles, list)
            else [product_smiles]
        )
        products = [
            mol_from_smiles_without_atom_maps(product)
            for product in product_values
            if product
        ]
        valid_products = [product for product in products if product]

        if not all(reactants) or not valid_products:
            return None

        reaction = AllChem.ChemicalReaction()
        for reactant in reactants:
            reaction.AddReactantTemplate(reactant)
        for product in valid_products:
            reaction.AddProductTemplate(product)

        return Draw.ReactionToImage(reaction, subImgSize=(350, 250), useSVG=False)
    except Exception as exc:
        print(f"Img Error: {exc}")
        return None
