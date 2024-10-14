# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import os

from peewee import BooleanField, CharField, ForeignKeyField, IntegerField, Model, SqliteDatabase, TextField

db_path = os.path.dirname(os.path.abspath(__file__)) + "/spacegroup.db"
db = SqliteDatabase(db_path)


class BaseModel(Model):
    """
    BaseModel class that serves as a base for other models in the application.

    Attributes:
        Meta (class): Inner class that defines the database configuration for the model.
    """

    class Meta:
        database = db


class SpaceGroup(BaseModel):
    """
    SpaceGroup class represents a crystallographic space group with various attributes.

    Attributes:
        full_symbol (CharField): The full symbol of the space group, unique.
        symbol (CharField): The abbreviated symbol of the space group.
        symmetry (CharField): The symmetry of the space group.
        point_group (CharField): The point group of the space group.
        patterson_symmetry (CharField): The Patterson symmetry of the space group.
        spacegroup_num (IntegerField): The space group number.
        hm_num (IntegerField): The Hermann-Mauguin number.
        wy_sets (IntegerField): The number of Wyckoff sets.
    """

    full_symbol = CharField(20, unique=True)
    symbol = CharField(20)
    symmetry = CharField(20)
    point_group = CharField(20)
    patterson_symmetry = CharField(20)
    spacegroup_num = IntegerField()
    hm_num = IntegerField()
    wy_sets = IntegerField()


class Wyckoff(BaseModel):
    """
    Wyckoff model representing a Wyckoff position in a crystallographic space group.

    Attributes:
        space_group (ForeignKeyField): A foreign key to the SpaceGroup model, representing the space group to which this Wyckoff position belongs.
        multiplicity (IntegerField): The multiplicity of the Wyckoff position.
        site_symmetry (CharField): The site symmetry of the Wyckoff position, limited to 20 characters.
        letter (CharField): The letter designation of the Wyckoff position, limited to 1 character.
        positions (TextField): The atomic positions associated with the Wyckoff position.
        reuse (BooleanField): A boolean indicating whether the Wyckoff position can be reused.
    """

    space_group = ForeignKeyField(SpaceGroup, backref="wyckoffs")
    multiplicity = IntegerField()
    site_symmetry = CharField(20)
    letter = CharField(1)
    positions = TextField()
    reuse = BooleanField()
    reuse = BooleanField()
