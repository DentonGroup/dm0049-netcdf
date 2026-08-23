# Copyright 1999-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8
PYTHON_COMPAT=( python3_{12..14} )
DISTUTILS_USE_PEP517=setuptools

inherit git-r3
inherit distutils-r1

DESCRIPTION="The netCDF acquisition container, and the one path into libnetcdf"
HOMEPAGE="https://github.com/DentonGroup/dm0049-netcdf"
EGIT_REPO_URI="https://github.com/DentonGroup/dm0049-netcdf.git"

LICENSE="ISC"
SLOT="0"
KEYWORDS=""

RDEPEND="
	dev-python/numpy[${PYTHON_USEDEP}]
	dev-python/pint[${PYTHON_USEDEP}]
	dev-python/pint-xarray[${PYTHON_USEDEP}]
	dev-python/xarray[${PYTHON_USEDEP}]
	dev-python/netcdf4[${PYTHON_USEDEP}]
"
DEPEND="${RDEPEND}"
