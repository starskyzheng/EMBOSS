/* @source needleall application
**
** Many-to-many pairwise alignment
**
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
******************************************************************************/




#include "emboss.h"
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <string.h>




/* @prog needleall ***********************************************************
**
** Many-to-many pairwise sequence alignment using Needleman-Wunsch algorithm
**
******************************************************************************/

// 存储序列信息的结构体
typedef struct {
    char* seq;
    char* name;
    ajuint len;
} SeqInfo;

// 处理命令行参数中的threads参数
void process_threads_arg(int* argc, char*** argv) {
    int i, j;
    int new_argc = *argc;
    char** new_argv = *argv;
    
    for(i = 1; i < *argc - 1; i++) {
        if(strcmp(new_argv[i], "-threads") == 0) {
            int num_threads = atoi(new_argv[i + 1]);
            if(num_threads > 0) {
                omp_set_num_threads(num_threads);
            }
            
            // 从参数列表中移除-threads及其值
            for(j = i; j < new_argc - 2; j++) {
                new_argv[j] = new_argv[j + 2];
            }
            new_argc -= 2;
            break;
        }
    }
    
    *argc = new_argc;
}

int main(int argc, char **argv)
{
    // 在embInit之前处理-threads参数
    process_threads_arg(&argc, &argv);

    AjPAlign align;
    AjPSeqall seqall;
    AjPSeqset seqset;
    const AjPSeq seqa;
    AjPSeq seqb;
    AjPStr alga;
    AjPStr algb;
    //AjPStr ss;
    AjPFile errorf;

    ajuint lena;
    ajuint lenb;
    ajuint k;
    ajuint i, j;

    const char *p;
    const char *q;

    ajint start1 = 0;
    ajint start2 = 0;

    ajint *compass;
    float* ix;
    float* iy;
    float* m;

    AjPMatrixf matrix;
    AjPSeqCvt cvt = 0;
    float **sub;

    float gapopen;
    float gapextend;
    float endgapopen;
    float endgapextend;
    float minscore;
    size_t maxarr = 1000;
    size_t len;

    float score;

    AjBool dobrief = ajTrue;
    AjBool endweight = ajFalse;

    float id   = 0.;
    float sim  = 0.;
    float idx  = 0.;
    float simx = 0.;

    AjPStr tmpstr = NULL;

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    embInit("needleall", argc, argv);

    matrix    = ajAcdGetMatrixf("datafile");
    seqset    = ajAcdGetSeqset("asequence");
    ajSeqsetTrim(seqset);
    seqall    = ajAcdGetSeqall("bsequence");
    gapopen   = ajAcdGetFloat("gapopen");
    gapextend = ajAcdGetFloat("gapextend");
    endgapopen   = ajAcdGetFloat("endopen");
    endgapextend = ajAcdGetFloat("endextend");
    minscore = ajAcdGetFloat("minscore");
    dobrief   = ajAcdGetBoolean("brief");
    endweight   = ajAcdGetBoolean("endweight");
    align     = ajAcdGetAlign("outfile");
    errorf    = ajAcdGetOutfile("errfile");

    gapopen = ajRoundFloat(gapopen, 8);
    gapextend = ajRoundFloat(gapextend, 8);

    sub = ajMatrixfGetMatrix(matrix);
    cvt = ajMatrixfGetCvt(matrix);

    // 收集所有序列信息
    ajuint seqset_size = ajSeqsetGetSize(seqset);
    SeqInfo* seqset_info = (SeqInfo*)malloc(seqset_size * sizeof(SeqInfo));
    size_t max_seq_len = 0;

    for(i = 0; i < seqset_size; i++) {
        seqa = ajSeqsetGetseqSeq(seqset, i);
        size_t curr_len = ajSeqGetLen(seqa);
        if(curr_len > max_seq_len) max_seq_len = curr_len;
        
        seqset_info[i].len = curr_len;
        seqset_info[i].seq = strdup(ajSeqGetSeqC(seqa));
        seqset_info[i].name = strdup(ajSeqGetNameC(seqa));
    }

    // 计算所需的最大数组大小
    maxarr = max_seq_len * max_seq_len;
    if(maxarr < 1000) maxarr = 1000;

    while(ajSeqallNext(seqall,&seqb))
    {
        ajSeqTrim(seqb);
        lenb = ajSeqGetLen(seqb);
        const char* seqb_seq = ajSeqGetSeqC(seqb);
        const char* seqb_name = ajSeqGetNameC(seqb);
        
        #pragma omp parallel private(compass,m,ix,iy,alga,algb,tmpstr,score,id,sim,idx,simx,start1,start2)
        {
            compass = NULL;
            m = NULL;
            ix = NULL;
            iy = NULL;
            alga = NULL;
            algb = NULL;
            tmpstr = NULL;
            
            AJCNEW0(compass, maxarr);
            AJCNEW0(m, maxarr);
            AJCNEW0(ix, maxarr);
            AJCNEW0(iy, maxarr);
            
            if(!compass || !m || !ix || !iy) {
                #pragma omp critical
                {
                    ajDie("Memory allocation failed in thread");
                }
            }
            
            alga = ajStrNewC("");
            algb = ajStrNewC("");
            tmpstr = ajStrNewC("");
            
            if(!alga || !algb || !tmpstr) {
                if(compass) AJFREE(compass);
                if(m) AJFREE(m);
                if(ix) AJFREE(ix);
                if(iy) AJFREE(iy);
                if(alga) ajStrDel(&alga);
                if(algb) ajStrDel(&algb);
                if(tmpstr) ajStrDel(&tmpstr);
                #pragma omp critical
                {
                    ajDie("String allocation failed in thread");
                }
            }

            #pragma omp for schedule(dynamic)
            for(i = 0; i < seqset_size; i++)
            {
                if(lenb > (LONG_MAX/(size_t)(seqset_info[i].len+1)))
                    continue;

                ajStrAssignC(&alga,"");
                ajStrAssignC(&algb,"");

                score = embAlignPathCalcWithEndGapPenalties(
                    seqset_info[i].seq, seqb_seq,
                    seqset_info[i].len, lenb,
                    gapopen, gapextend, endgapopen, endgapextend,
                    &start1, &start2, sub, cvt, m, ix, iy,
                    compass, ajFalse, endweight);

                if(score > minscore) {
                    embAlignWalkNWMatrixUsingCompass(
                        seqset_info[i].seq, seqb_seq,
                        &alga, &algb,
                        seqset_info[i].len, lenb,
                        &start1, &start2, compass);

                    omp_set_lock(&writelock);
                    
                    if(!ajAlignFormatShowsSequences(align))
                    {
                        ajAlignDefineCC(align, ajStrGetPtr(alga),
                                ajStrGetPtr(algb),
                                seqset_info[i].name,
                                seqb_name);
                        ajAlignSetScoreR(align, score);
                    }
                    else
                    {
                        const AjPSeq curr_seqa = ajSeqsetGetseqSeq(seqset, i);
                        embAlignReportGlobal(align, curr_seqa, seqb,
                                alga, algb,
                                start1, start2,
                                gapopen, gapextend,
                                score, matrix,
                                0, 0);
                    }

                    if(!dobrief)
                    {
                        embAlignCalcSimilarity(alga, algb,
                                sub, cvt, seqset_info[i].len, lenb,
                                &id, &sim, &idx, &simx);
                        ajStrAssignC(&tmpstr, "");
                        ajFmtPrintS(&tmpstr,
                                "Longest_Identity = %5.2f%%\n",
                                id);
                        ajFmtPrintAppS(&tmpstr,
                                "Longest_Similarity = %5.2f%%\n",
                                sim);
                        ajFmtPrintAppS(&tmpstr,
                                "Shortest_Identity = %5.2f%%\n",
                                idx);
                        ajFmtPrintAppS(&tmpstr,
                                "Shortest_Similarity = %5.2f%%",
                                simx);
                        ajAlignSetSubHeaderApp(align, tmpstr);
                    }
                    
                    ajAlignWrite(align);
                    ajAlignReset(align);
                    
                    omp_unset_lock(&writelock);
                }
                else {
                    omp_set_lock(&writelock);
                    ajFmtPrintF(errorf,
                            "Alignment score (%.1f) is less than minimum score"
                            "(%.1f) for sequences %s vs %s\n",
                            score, minscore,
                            seqset_info[i].name,
                            seqb_name);
                    omp_unset_lock(&writelock);
                }
            }
            
            if(compass) AJFREE(compass);
            if(m) AJFREE(m);
            if(ix) AJFREE(ix);
            if(iy) AJFREE(iy);
            if(alga) ajStrDel(&alga);
            if(algb) ajStrDel(&algb);
            if(tmpstr) ajStrDel(&tmpstr);
        }
    }

    // 清理序列信息
    for(i = 0; i < seqset_size; i++) {
        free(seqset_info[i].seq);
        free(seqset_info[i].name);
    }
    free(seqset_info);

    omp_destroy_lock(&writelock);

    if(!ajAlignFormatShowsSequences(align))
    {
        ajMatrixfDel(&matrix);        
    }

    ajAlignClose(align);
    ajAlignDel(&align);
    ajFileClose(&errorf);

    ajSeqallDel(&seqall);
    ajSeqsetDel(&seqset);
    ajSeqDel(&seqb);
    
    //ajStrDel(&ss);

    embExit();

    return 0;
}

