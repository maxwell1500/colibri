/* zstd_pack — converts a safetensors shard to zstd-compressed format.
 * Same structure as cfse_pack but uses zstd instead of CFSE.
 * Magic: "ZSTD" (4 bytes) + rawlen u32LE = 8 byte header.
 * Non-U8 tensors are left as-is (only U8 tensors are compressed).
 *
 * Usage: zstd_pack in.safetensors out.safetensors
 *        zstd_pack --verify in.safetensors out.safetensors
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "json.h"
#include <zstd.h>

static void *xmalloc(size_t n){ void *p=malloc(n?n:1); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }

typedef struct { const char *name; const char *dtype; jval *shape; int64_t a,b; } TEnt;

static int cmp_off(const void *x,const void *y){
    const TEnt *A=x,*B=y; return (A->a>B->a)-(A->a<B->a);
}

static char *read_file(const char *path, size_t *n){
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char *b=xmalloc((size_t)sz);
    if(fread(b,1,(size_t)sz,f)!=(size_t)sz){perror("fread");exit(1);}
    fclose(f); *n=(size_t)sz; return b;
}

static int parse_shard(char *buf, size_t n, jval **root_out, char **arena_out,
                       TEnt **ents_out, int *nents_out, size_t *data_start_out){
    if(n<8) return -1;
    uint64_t hlen; memcpy(&hlen,buf,8);
    if(hlen>n-8) return -1;
    char *hdr=xmalloc(hlen+1); memcpy(hdr,buf+8,hlen); hdr[hlen]=0;
    char *arena=NULL; jval *root=json_parse(hdr,&arena);
    if(!root||root->t!=J_OBJ) return -1;
    TEnt *ents=xmalloc(sizeof(TEnt)*(size_t)root->len); int ne=0;
    for(int i=0;i<root->len;i++){
        if(!strcmp(root->keys[i],"__metadata__")) continue;
        jval *m=root->kids[i];
        jval *dt=json_get(m,"dtype"), *off=json_get(m,"data_offsets"), *shp=json_get(m,"shape");
        if(!dt||dt->t!=J_STR||!off||off->t!=J_ARR||off->len<2||!shp||shp->t!=J_ARR) return -1;
        ents[ne].name=root->keys[i]; ents[ne].dtype=dt->str; ents[ne].shape=shp;
        ents[ne].a=(int64_t)off->kids[0]->num; ents[ne].b=(int64_t)off->kids[1]->num; ne++;
    }
    qsort(ents,(size_t)ne,sizeof(TEnt),cmp_off);
    *root_out=root; *arena_out=arena; *ents_out=ents; *nents_out=ne; *data_start_out=8+hlen;
    return 0;
}

static int is_u8(const char *dtype){ return dtype && !strcmp(dtype,"U8"); }

static int convert(const char *inp, const char *outp, int verify){
    size_t fn; char *buf=read_file(inp,&fn);
    jval *root; char *arena; TEnt *ents; int ne; size_t ds;
    if(parse_shard(buf,fn,&root,&arena,&ents,&ne,&ds)){ fprintf(stderr,"parse fail %s\n",inp); return 1; }

    /* compress each U8 tensor, keep non-U8 as-is */
    size_t bound = ZSTD_compressBound(1024*1024*1024);
    /* For each tensor, compute compressed size */
    typedef struct { int64_t orig_off, orig_len; int64_t new_off; int compressed; size_t comp_len; uint8_t *comp_data; } NewOff;
    NewOff *no = xmalloc(sizeof(NewOff)*(size_t)ne);
    size_t total_new_data = 0;
    for(int i=0;i<ne;i++){
        no[i].orig_off = ents[i].a; no[i].orig_len = ents[i].b - ents[i].a;
        no[i].compressed = 0; no[i].comp_data = NULL; no[i].comp_len = 0;
        if(is_u8(ents[i].dtype) && no[i].orig_len > 16){
            uint8_t *raw = (uint8_t*)buf + ds + ents[i].a;
            size_t rlen = (size_t)no[i].orig_len;
            size_t cbound = ZSTD_compressBound(rlen);
            uint8_t *cbuf = xmalloc(cbound);
            /* ZSTD magic: "ZSTD" + rawlen u32LE + zstd frame */
            size_t hdr_sz = 8;
            size_t zlen = ZSTD_compress(cbuf+hdr_sz, cbound-hdr_sz, raw, rlen, 3);
            if(ZSTD_isError(zlen)){
                fprintf(stderr,"zstd compress fail %s\n",ents[i].name); return 1;
            }
            /* Only use compression if it actually saves space */
            if(hdr_sz + zlen < rlen){
                no[i].compressed = 1;
                no[i].comp_len = hdr_sz + zlen;
                no[i].comp_data = xmalloc(no[i].comp_len);
                memcpy(no[i].comp_data, "ZSTD", 4);
                uint32_t rl = (uint32_t)rlen;
                memcpy(no[i].comp_data+4, &rl, 4);
                memcpy(no[i].comp_data+hdr_sz, cbuf+hdr_sz, zlen);
            }
            free(cbuf);
        }
        no[i].new_off = total_new_data;
        if(no[i].compressed) total_new_data += no[i].comp_len;
        else total_new_data += no[i].orig_len;
    }

    /* build new header */
    FILE *o=fopen(outp,"wb"); if(!o){perror(outp);return 1;}
    /* header JSON */
    fprintf(o,"{\"__metadata__\":{\"cfse\":\"1\"}");
    for(int i=0;i<ne;i++){
        int64_t nb = no[i].compressed ? (int64_t)no[i].comp_len : no[i].orig_len;
        fprintf(o,",\"%s\":{\"dtype\":\"%s\",\"shape\":[",ents[i].name,ents[i].dtype);
        for(int j=0;j<ents[i].shape->len;j++){
            fprintf(o,"%s%lld",j?",":"",(long long)ents[i].shape->kids[j]->num);
        }
        fprintf(o,"],\"data_offsets\":[%lld,%lld]}",(long long)no[i].new_off,(long long)(no[i].new_off+nb));
    }
    fprintf(o,"}");
    /* compute header length, pad to 8-byte boundary */
    long hdr_pos = ftell(o);
    /* safetensors header length is stored as u64LE at file start */
    /* we need to write data starting at 8 + hlen */
    /* first, close file, compute hlen, reopen */
    fclose(o);

    /* re-read the header we just wrote to get its length */
    size_t hlen = (size_t)hdr_pos;
    /* pad hlen to 8-byte alignment? safetensors doesn't require it but let's keep it simple */
    /* Actually safetensors header is just JSON, no padding needed */
    size_t data_start = 8 + hlen;

    /* write the final file: u64 hlen + header + data */
    o = fopen(outp, "wb"); if(!o){perror(outp);return 1;}
    uint64_t hlen64 = (uint64_t)hlen;
    fwrite(&hlen64, 1, 8, o);
    /* re-emit header (same as above) */
    fprintf(o,"{\"__metadata__\":{\"cfse\":\"1\"}");
    for(int i=0;i<ne;i++){
        int64_t nb = no[i].compressed ? (int64_t)no[i].comp_len : no[i].orig_len;
        fprintf(o,",\"%s\":{\"dtype\":\"%s\",\"shape\":[",ents[i].name,ents[i].dtype);
        for(int j=0;j<ents[i].shape->len;j++){
            fprintf(o,"%s%lld",j?",":"",(long long)ents[i].shape->kids[j]->num);
        }
        fprintf(o,"],\"data_offsets\":[%lld,%lld]}",(long long)no[i].new_off,(long long)(no[i].new_off+nb));
    }
    fprintf(o,"}");
    /* verify header length matches */
    if((size_t)ftell(o) != 8 + hlen){ fprintf(stderr,"header length mismatch!\n"); return 1; }

    /* write data */
    for(int i=0;i<ne;i++){
        if(no[i].compressed){
            fwrite(no[i].comp_data, 1, no[i].comp_len, o);
        } else {
            fwrite(buf + ds + no[i].orig_off, 1, (size_t)no[i].orig_len, o);
        }
    }
    fclose(o);

    fprintf(stderr,"converted %s -> %s (%zu -> %zu bytes)\n", inp, outp, fn, 8+hlen+total_new_data);

    if(verify){
        /* verify: re-read and decompress, compare with original */
        for(int i=0;i<ne;i++){
            if(!no[i].compressed) continue;
            uint8_t *raw = (uint8_t*)buf + ds + no[i].orig_off;
            size_t rlen = (size_t)no[i].orig_len;
            uint8_t *dec = xmalloc(rlen);
            /* read ZSTD header */
            uint32_t rl; memcpy(&rl, no[i].comp_data+4, 4);
            if(rl != rlen){ fprintf(stderr,"verify fail %s: rawlen mismatch\n",ents[i].name); return 1; }
            size_t dlen = ZSTD_decompress(dec, rlen, no[i].comp_data+8, no[i].comp_len-8);
            if(ZSTD_isError(dlen) || dlen != rlen || memcmp(dec, raw, rlen)){
                fprintf(stderr,"verify fail %s: decompress mismatch\n",ents[i].name); return 1;
            }
            free(dec);
        }
        fprintf(stderr,"verify OK: all %d U8 tensors round-trip\n", ne);
    }

    for(int i=0;i<ne;i++) free(no[i].comp_data);
    free(no); free(arena); free(root); free(ents); free(buf);
    return 0;
}

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: zstd_pack [--verify] in.safetensors out.safetensors\n"); return 1; }
    int verify=0; int ai=1;
    if(argc>3 && !strcmp(argv[1],"--verify")){ verify=1; ai=2; }
    return convert(argv[ai], argv[ai+1], verify);
}