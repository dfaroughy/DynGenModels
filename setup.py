
import setuptools

setuptools.setup(name="DynGenModels",
                version=1.0,
                url="git@github.com:dfaroughy/DynGenModels.git",
                packages=setuptools.find_packages("src"),
                package_dir={"": "src"}
                )