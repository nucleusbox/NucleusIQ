import setuptools
from pathlib import Path

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="nucleusiq",
    version="0.1.0",
    author="Nucleusbox",
    author_email="info@nucleusbox.com",
    description="NucleusIQ is an open-source framework for building and managing autonomous AI agents. It offers diverse strategies and architectures to create intelligent chatbots, financial tools, and multi-agent systems. With NucleusIQ, developers have the core components and flexibility needed to develop advanced AI applications effortlessly.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nucleusbox/NucleusIQ",
    project_urls={
        "Documentation": "https://github.com/nucleusbox/NucleusIQ#readme",
        "Source": "https://github.com/nucleusbox/NucleusIQ",
        "Tracker": "https://github.com/nucleusbox/NucleusIQ/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6,<4.0",
    install_requires=[
        "requests>=2.25.1",
        "flask>=2.0.1",
        # Add other runtime dependencies here
    ],
    include_package_data=True,
)
