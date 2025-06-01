"use client";

import React from "react";

import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import { Download, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";

import { Input } from "@/components/ui/input";

import { getHttpErrorMessage } from "@/lib/http";

import { useVideoInfo } from "@/services/api/queries";

const formSchema = z.object({
  postUrl: z.string().url({
    message: "Provide a valid Instagram post link",
  }),
});

export function InstagramVideoForm() {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      postUrl: "",
    },
  });

  const { error, isPending, mutateAsync: getVideoInfo } = useVideoInfo();

  const httpError = getHttpErrorMessage(error);

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const { postUrl } = values;
    try {
      console.log("getting video info", postUrl);
      const videoInfo = await getVideoInfo({ postUrl });
  
      const { filename, videoUrl } = videoInfo;
  
      console.log("videoUrl:", videoUrl);
  
      await downloadFile(videoUrl, filename);
    } catch (error: any) {
      console.log(error);
    }
  }
  
  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="bg-accent/20 my-4 flex w-full max-w-2xl flex-col items-center rounded-lg border px-4 pb-16 pt-8 shadow-md sm:px-8"
      >
        <div className="mb-2 h-6 w-full px-2 text-start text-red-500">
          {httpError}
        </div>
        <div className="relative mb-6 flex w-full flex-col items-center gap-4 sm:flex-row">
          <FormField
            control={form.control}
            name="postUrl"
            render={({ field }) => (
              <FormItem className="w-full">
                <FormControl>
                  <Input
                    disabled={isPending}
                    type="url"
                    placeholder="Paste your Instagram link here..."
                    className="h-12 w-full sm:pr-28"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button
            disabled={isPending}
            type="submit"
            className="right-1 top-1 w-full sm:absolute sm:w-fit"
          >
            {isPending ? (
              <Loader2 className="mr-2 animate-spin" />
            ) : (
              <Download className="mr-2" />
            )}
            Download
          </Button>
        </div>
        <p className="text-muted-foreground text-center text-xs">
          If the download opens a new page, right click the video and then click{" "}
          Save as video.
        </p>
      </form>
    </Form>
  );
}

// Utility function for download
export async function downloadFile(videoUrl: string, filename: string) {
  try {
    const response = await fetch(videoUrl);

    if (!response.ok) {
      throw new Error("Failed to fetch the video for download.");
    }

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filename; // Set the filename for the download
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // Cleanup blob URL
    window.URL.revokeObjectURL(blobUrl);
  } catch (error) {
    console.error("Error during file download:", error);
  }
}
