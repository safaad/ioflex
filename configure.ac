define([version_major], 0)
define([version_minor], 0)
define([version_revision], 1)



AC_DEFINE_UNQUOTED(LIBIOFLEX_MAJOR, ${version_major},[ioflex Major Version])
AC_SUBST(LIBIOFLEX_MAJOR)
AC_DEFINE_UNQUOTED(LIBIOFLEX_MINOR, ${version_minor},[ioflex Minor Version])
AC_SUBST(LIBIOFLEX_MINOR)
AC_DEFINE_UNQUOTED(LIBIOFLEX_REVISION, ${version_revision}, [ioflex Revision])
AC_SUBST(LIBIOFLEX_REVISION)


AC_INIT(libioflex,[version_major.version_minor.version_revision],safadiaby9@gmail.com)

AC_LANG([C])
AM_INIT_AUTOMAKE([subdir-objects])
AC_PROG_CC([mpicc])
CC=mpicc

AC_USE_SYSTEM_EXTENSIONS([_GNU_SOURCE])
AC_CONFIG_FILES([Makefile
                src/Makefile])



# Check for libconfuse
AC_ARG_WITH([libconfuse],
            [AC_HELP_STRING([--with-libconfuse=DIR],
                            [Use libconfuse library [default=yes]])],,
            [withval=yes])
case "$withval" in
    yes)
        HAVE_CFG="yes"
        AC_CHECK_HEADERS([confuse.h], [], [unset HAVE_CFG], 
        [[#ifdef HAVE_CFG_H
          # include<confuse.h>
          #endif
        ]])

        AC_CHECK_LIB([confuse], [cfg_init],, [unset HAVE_CFG])

        if test -z "$HAVE_CFG"; then
            AC_MSG_ERROR([could not find libconfuse])
        fi
        ;;
    no)
        AC_MSG_ERROR([libconfuse is required to build IOFlex])
        ;;
    *)
        HAVE_CFG="yes"
        case "$withval" in
            *,*)
                cfg_inc="`echo $withval | cut -f1 -d,`"
                cfg_lib="`echo $withval | cut -f1 -d, -s`"
                ;;
            *)
                if test -n "$withval"; then
                    cfg_inc="$withval/include"
                    cfg_lib="$withval/lib"
                fi
                ;;
        esac
        if test "X$cfg_inc" = "X/usr/include"; then
            cfg_inc=""
        fi
        if test "X$cfg_lib" = "X/usr/lib"; then
            cfg_lib=""
        fi

        if test -n "$cfg_inc"; then
            CFLAGS="$CFLAGS -I$cfg_inc"
        fi

        AC_CHECK_HEADERS([confuse.h], [], [unset HAVE_CFG])
        
        if test -n "$cfg_lib"; then
            LDFLAGS="$LDFLAGS -L$cfg_lib"
        fi

        AC_CHECK_LIB([confuse], [cfg_init],, [unset HAVE_CFG])

        if test -z "$HAVE_CFG"; then
            AC_MSG_ERROR([could not find libconfuse])
        fi
        ;;
    esac


AC_PREFIX_DEFAULT([`pwd`])
LT_INIT([disable-static])
AC_CHECK_LIB([dl], [dlsym], [], [AC_MSG_ERROR(libld is not installed)])

AC_OUTPUT